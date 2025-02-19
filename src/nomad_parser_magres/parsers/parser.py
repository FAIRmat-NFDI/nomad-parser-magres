import json
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad_simulations.schema_packages.model_system import Cell
    from structlog.stdlib import BoundLogger

from typing import Dict

from nomad.app.v1.models.models import MetadataRequired
from nomad.config import config
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.workflow import Link, TaskReference
from nomad.parsing import MatchingParser
from nomad.parsing.file_parser import Quantity, TextParser
from nomad.search import search
from nomad.units import ureg
from nomad.utils import extract_section
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program
from nomad_simulations.schema_packages.model_method import (
    DFT,
    ModelMethod,
    XCFunctional,
)
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem
from nomad_simulations.schema_packages.numerical_settings import KMesh, KSpace

# utility function used to get auxiliary files next to the `mainfile`
from nomad_parser_magres.parsers.utils import get_files
from nomad_parser_magres.schema_packages.ccpnc_metadata import (
    ORCID,
    CCPNCMetadata,
    CCPNCRecord,
    ExternalDatabaseReference,
    FreeTextMetadata,
    MaterialProperties,
)
from nomad_parser_magres.schema_packages.package import CCPNCSimulation as Simulation
from nomad_parser_magres.schema_packages.package import (
    ElectricFieldGradient,
    ElectricFieldGradients,
    MagneticShieldingTensor,
    MagneticSusceptibility,
    Outputs,
    SpinSpinCoupling,
)
from nomad_parser_magres.schema_packages.workflow import (
    NMRMagRes,
    NMRMagResMethod,
    NMRMagResResults,
)

re_float = r" *[-+]?\d+\.\d*(?:[Ee][-+]\d+)? *"

configuration = config.get_plugin_entry_point(
    "nomad_parser_magres.parsers:nomad_parser_magres_plugin"
)


class MagresFileParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        self._quantities = [
            Quantity("lattice_units", r"units *lattice *([a-zA-Z]+)"),
            Quantity("atom_units", r"units *atom *([a-zA-Z]+)"),
            Quantity("ms_units", r"units *ms *([a-zA-Z]+)"),
            Quantity("efg_units", r"units *efg *([a-zA-Z]+)"),
            Quantity("efg_local_units", r"units *efg_local *([a-zA-Z]+)"),
            Quantity("efg_nonlocal_units", r"units *efg_nonlocal *([a-zA-Z]+)"),
            Quantity("isc_units", r"units *isc *([a-zA-Z\^\d\.\-]+)"),
            Quantity("isc_fc_units", r"units *isc_fc *([a-zA-Z\^\d\.\-]+)"),
            Quantity("isc_spin_units", r"units *isc_spin *([a-zA-Z\^\d\.\-]+)"),
            Quantity(
                "isc_orbital_p_units", r"units *isc_orbital_p *([a-zA-Z\^\d\.\-]+)"
            ),
            Quantity(
                "isc_orbital_d_units", r"units *isc_orbital_d *([a-zA-Z\^\d\.\-]+)"
            ),
            Quantity("sus_units", r"units *sus *([a-zA-Z\^\d\.\-]+)"),
            Quantity("cutoffenergy_units", r"units *calc\_cutoffenergy *([a-zA-Z]+)"),
            Quantity(
                "calculation",
                r"([\[\<]*calculation[\>\]]*[\s\S]+?)(?:[\[\<]*\/calculation[\>\]]*)",
                sub_parser=TextParser(
                    quantities=[
                        Quantity("code", r"calc\_code *([a-zA-Z]+)"),
                        Quantity(
                            "code_version", r"calc\_code\_version *([a-zA-Z\d\.]+)"
                        ),
                        Quantity(
                            "code_hgversion",
                            r"calc\_code\_hgversion ([a-zA-Z\d\:\+\s]*)\n",
                            flatten=False,
                        ),
                        Quantity(
                            "code_platform", r"calc\_code\_platform *([a-zA-Z\d\_]+)"
                        ),
                        Quantity("name", r"calc\_name *([\w]+)"),
                        Quantity("comment", r"calc\_comment *([\w]+)"),
                        Quantity("xcfunctional", r"calc\_xcfunctional *([\w]+)"),
                        Quantity(
                            "cutoffenergy",
                            rf"calc\_cutoffenergy({re_float})(?P<__unit>\w+)",
                        ),
                        Quantity(
                            "pspot",
                            r"calc\_pspot *([\w]+) *([\w\.\|\(\)\=\:]+)",
                            repeats=True,
                        ),
                        Quantity(
                            "kpoint_mp_grid",
                            r"calc\_kpoint\_mp\_grid *([\w]+) *([\w]+) *([\w]+)",
                        ),
                        Quantity(
                            "kpoint_mp_offset",
                            rf"calc\_kpoint\_mp\_offset({re_float * 3})$",
                        ),
                    ]
                ),
            ),
            Quantity(
                "atoms",
                r"([\[\<]*atoms[\>\]]*[\s\S]+?)(?:[\[\<]*\/atoms[\>\]]*)",
                sub_parser=TextParser(
                    quantities=[
                        Quantity("lattice", rf"lattice({re_float * 9})"),
                        Quantity("symmetry", r"symmetry *([\w\-\+\,]+)", repeats=True),
                        Quantity(
                            "atom",
                            rf"atom *([a-zA-Z]+) *[a-zA-Z\d]* *([\d]+) *({re_float * 3})",
                            repeats=True,
                        ),
                    ]
                ),
            ),
            Quantity(
                "magres",
                r"([\[\<]*magres[\>\]]*[\s\S]+?)(?:[\[\<]*\/magres[\>\]]*)",
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            "ms", rf"ms *(\w+) *(\d+)({re_float * 9})", repeats=True
                        ),
                        Quantity(
                            "efg", rf"efg *(\w+) *(\d+)({re_float * 9})", repeats=True
                        ),
                        Quantity(
                            "efg_local",
                            rf"efg_local *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity(
                            "efg_nonlocal",
                            rf"efg_nonlocal *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity(
                            "isc",
                            rf"isc *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity(
                            "isc_fc",
                            rf"isc_fc *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity(
                            "isc_orbital_p",
                            rf"isc_orbital_p *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity(
                            "isc_orbital_d",
                            rf"isc_orbital_d *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity(
                            "isc_spin",
                            rf"isc_spin *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})",
                            repeats=True,
                        ),
                        Quantity("sus", rf"sus *({re_float * 9})", repeats=True),
                    ]
                ),
            ),
        ]


class MagresParser(MatchingParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magres_file_parser = MagresFileParser()

        self._xc_functional_map = {
            "LDA": ["LDA_C_PZ", "LDA_X_PZ"],
            "PW91": ["GGA_C_PW91", "GGA_X_PW91"],
            "PBE": ["GGA_C_PBE", "GGA_X_PBE"],
            "RPBE": ["GGA_X_RPBE"],
            "WC": ["GGA_C_PBE_GGA_X_WC"],
            "PBESOL": ["GGA_X_RPBE"],
            "BLYP": ["GGA_C_LYP", "LDA_X_B88"],
            "B3LYP": ["HYB_GGA_XC_B3LYP5"],
            "HF": ["HF_X"],
            "HF-LDA": ["HF_X_LDA_C_PW"],
            "PBE0": ["HYB_GGA_XC_PBEH"],
            "HSE03": ["HYB_GGA_XC_HSE03"],
            "HSE06": ["HYB_GGA_XC_HSE06"],
            "RSCAN": ["MGGA_X_RSCAN", "MGGA_C_RSCAN"],
        }

    def _check_units_magres(self, logger: "BoundLogger") -> None:
        """
        Check if the units of the NMR quantities are magres standard. If not, a warning
        is issued and the default units are used.
        """
        allowed_units = {
            "lattice": "Angstrom",
            "atom": "Angstrom",
            "ms": "ppm",
            "efg": "au",
            "efg_local": "au",
            "efg_nonlocal": "au",
            "isc": "10^19.T^2.J^-1",
            "isc_fc": "10^19.T^2.J^-1",
            "isc_orbital_p": "10^19.T^2.J^-1",
            "isc_orbital_d": "10^19.T^2.J^-1",
            "isc_spin": "10^19.T^2.J^-1",
            "sus": "10^-6.cm^3.mol^-1",
        }
        for key, value in allowed_units.items():
            data = self.magres_file_parser.get(f"{key}_units", "")
            if data and data != value:
                logger.warning(
                    "The units of the NMR quantities are not parsed if they are not magres standard. "
                    "We will use the default units.",
                    data={
                        "quantities": key,
                        "standard_units": value,
                        "parsed_units": data,
                    },
                )

    def init_parser(self, logger: "BoundLogger") -> None:
        """
        Initialize the `MagresFileParser` with the mainfile and logger.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        self.magres_file_parser.mainfile = self.mainfile
        self.magres_file_parser.logger = logger

    def parse_atomic_cell(
        self, atoms: Optional[TextParser], logger: "BoundLogger"
    ) -> Optional[AtomicCell]:
        """
        Parse the `AtomicCell` section from the magres file.

        Args:
            atoms (Optional[TextParser]): The parsed text section [atoms][/atoms] of the magres file.
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[AtomicCell]: The parsed `AtomicCell` section.
        """
        # Check if [atoms][/atoms] was correctly parsed
        if not atoms:
            logger.warning("Could not find atomic structure in magres file.")
            return None
        atomic_cell = AtomicCell()

        # Parse `lattice_vectors` and `periodic_boundary_conditions`
        try:
            lattice_vectors = np.reshape(np.array(atoms.get("lattice", [])), (3, 3))
            atomic_cell.lattice_vectors = lattice_vectors * ureg.angstrom
            pbc = (
                [True, True, True]
                if lattice_vectors is not None
                else [False, False, False]
            )
            atomic_cell.periodic_boundary_conditions = pbc
        except Exception:
            logger.warning(
                "Could not parse `lattice_vectors` and `periodic_boundary_conditions`."
            )
            return None

        # Parse `positions` and `AtomsState` list
        atoms_list = atoms.get("atom", [])
        if len(atoms_list) == 0:
            logger.warning(
                "Could not find atom `positions` and their chemical symbols in magres file."
            )
            return None
        positions = []
        atoms_states = []
        for atom in atoms_list:
            atoms_states.append(AtomsState(chemical_symbol=atom[0]))
            positions.append(atom[2:])
        atomic_cell.positions = positions * ureg.angstrom
        atomic_cell.atoms_state = atoms_states
        return atomic_cell

    def parse_model_system(self, logger: "BoundLogger") -> Optional[ModelSystem]:
        """
        Parse the `ModelSystem` section from the magres file if the [atoms][/atoms] section
        in the magres file was correctly matched.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[ModelSystem]: The parsed `ModelSystem` section.
        """
        # Check if [atoms][/atoms] was correctly parsed
        atoms = self.magres_file_parser.get("atoms")
        if not atoms:
            logger.warning("Could not find atomic structure in magres file.")
            return None

        # Parse `ModelSystem` and its `cell`
        model_system = ModelSystem()
        model_system.is_representative = True
        atomic_cell = self.parse_atomic_cell(atoms=atoms, logger=logger)
        model_system.cell.append(atomic_cell)
        return model_system

    def parse_xc_functional(
        self, calculation_params: Optional[TextParser]
    ) -> list[XCFunctional]:
        """
        Parse the exchange-correlation functional information from the magres file. This
        uses the `libxc` naming convention.

        Args:
            calculation_params (Optional[TextParser]): The parsed [calculation][/calculation] block parameters.

        Returns:
            list[XCFunctional]: The parsed `XCFunctional` sections.
        """
        xc_functional = calculation_params.get("xcfunctional", "LDA")
        xc_functional_labels = self._xc_functional_map.get(xc_functional, [])
        xc_sections = []
        for xc in xc_functional_labels:
            functional = XCFunctional(libxc_name=xc)
            if "_X_" in xc:
                functional.name = "exchange"
            elif "_C_" in xc:
                functional.name = "correlation"
            elif "HYB" in xc:
                functional.name = "hybrid"
            else:
                functional.name = "contribution"
            xc_sections.append(functional)
        return xc_sections

    def parse_model_method(
        self, calculation_params: Optional[TextParser]
    ) -> ModelMethod:
        """
        Parse the `ModelMethod` section by extracting information about the NMR method: basis set,
        exchange-correlation functional, cutoff energy, and K mesh.

        Note: only CASTEP-like method parameters are currently being supported.

        Args:
            calculation_params (Optional[TextParser]): The parsed [calculation][/calculation] block parameters.

        Returns:
            Optional[ModelMethod]: The parsed `ModelMethod` section.
        """
        model_method = DFT(name="NMR")

        # Parse `XCFunctinals` information
        xc_functionals = self.parse_xc_functional(calculation_params=calculation_params)
        if len(xc_functionals) > 0:
            model_method.xc_functionals = xc_functionals

        # TODO add when @ndaelman-hu finishes implementation of `BasisSet`
        # # Basis set parsing (adding cutoff energies units check)
        # cutoff = calculation_params.get('cutoffenergy')
        # if cutoff.dimensionless:
        #     cutoff_units = self.magres_file_parser.get('cutoffenergy_units', 'eV')
        #     if cutoff_units == 'Hartree':
        #         cutoff_units = 'hartree'
        #     cutoff = cutoff.magnitude * ureg(cutoff_units)
        # sec_basis_set = BasisSetContainer(
        #     type='plane waves',
        #     scope=['wavefunction'],
        #     basis_set=[BasisSet(scope=['valence'], type='plane waves', cutoff=cutoff)],
        # )
        # sec_method.electrons_representation.append(sec_basis_set)

        # Parse `KSpace` as a `NumericalSettings` section
        k_mesh = KMesh(
            grid=calculation_params.get("kpoint_mp_grid", [1, 1, 1]),
            offset=calculation_params.get("kpoint_mp_offset", [0, 0, 0]),
        )
        model_method.numerical_settings.append(KSpace(k_mesh=[k_mesh]))

        return model_method

    def parse_magnetic_shieldings(
        self, magres_data: TextParser, cell: "Cell", logger: "BoundLogger"
    ) -> list[MagneticShieldingTensor]:
        """
        Parse the magnetic shieldings from the magres file and assign `entity_ref` to the specific `AtomsState`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell (Cell): The parsed `Cell` section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[MagneticShieldingTensor]: The list of parsed `MagneticShieldingTensor` sections.
        """
        n_atoms = len(cell.atoms_state)
        data = magres_data.get("ms", [])

        # Initial check on the size of the matched text
        if np.size(data) != n_atoms * (9 + 2):  # 2 extra columns with atom labels
            logger.warning(
                "The shape of the matched text from the magres file for the `ms` does not coincide with the number of atoms."
            )
            return []

        # Parse magnetic shieldings and their refs to the specific `AtomsState`
        magnetic_shieldings = []
        for i, atom_data in enumerate(data):
            # values = np.transpose(np.reshape(atom_data[2:], (3, 3)))
            values = np.reshape(atom_data[2:], (3, 3))  # No need to transpose
            sec_ms = MagneticShieldingTensor(entity_ref=cell.atoms_state[i])
            sec_ms.value = values * 1e-6 * ureg("dimensionless")
            magnetic_shieldings.append(sec_ms)
        return magnetic_shieldings

    def parse_electric_field_gradients(
        self, magres_data: TextParser, cell: "Cell", logger: "BoundLogger"
    ) -> ElectricFieldGradients:
        """
        Parse the electric field gradients from the magres file and assign `entity_ref` to the specific `AtomsState`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell (Cell): The parsed `Cell` section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            ElectricFieldGradients: The parsed `ElectricFieldGradients` section.
        """
        n_atoms = len(cell.atoms_state)
        efg_contributions = {
            "efg_local": "local",
            "efg_nonlocal": "non_local",
            "efg": "total",
        }
        # electric_field_gradients = []
        electric_field_gradients = ElectricFieldGradients()
        for tag, contribution in efg_contributions.items():
            data = magres_data.get(tag, [])

            # Initial check on the size of the matched text
            if np.size(data) != n_atoms * (9 + 2):  # 2 extra columns with atom labels
                logger.warning(
                    "The shape of the matched text from the magres file for the `efg` does not coincide with the number of atoms."
                )
                # return []
                continue  # Log a warning and continue processing the remaining tags

            # Parse electronic field gradients for each contribution and their refs to the specific `AtomsState`
            for i, atom_data in enumerate(data):
                # values = np.transpose(np.reshape(atom_data[2:], (3, 3)))
                values = np.reshape(atom_data[2:], (3, 3))  # no need to transpose
                sec_efg = ElectricFieldGradient(
                    type=contribution, entity_ref=cell.atoms_state[i]
                )
                sec_efg.value = values * 9.717362e21 * ureg("V/m^2")
                # electric_field_gradients.append(sec_efg)
                if contribution == "total":
                    electric_field_gradients.efg_total.append(sec_efg)
                elif contribution == "local":
                    electric_field_gradients.efg_local.append(sec_efg)
                elif contribution == "non_local":
                    electric_field_gradients.efg_nonlocal.append(sec_efg)
        return electric_field_gradients

    def parse_spin_spin_couplings(
        self, magres_data: TextParser, cell: "Cell", logger: "BoundLogger"
    ) -> list[SpinSpinCoupling]:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `AtomsState`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell (Cell): The parsed `Cell` section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[SpinSpinCoupling]: The list of parsed `SpinSpinCoupling` sections.
        """
        n_atoms = len(cell.atoms_state)
        isc_contributions = {
            "isc_fc": "fermi_contact",
            "isc_orbital_p": "orbital_paramagnetic",
            "isc_orbital_d": "orbital_diamagnetic",
            "isc_spin": "spin_dipolar",
            "isc": "total",
        }
        spin_spin_couplings = []
        for tag, contribution in isc_contributions.items():
            data = magres_data.get(tag, [])

            # Initial check on the size of the matched text
            if np.size(data) != n_atoms**2 * (
                9 + 4
            ):  # 4 extra columns with atom labels
                logger.warning(
                    "The shape of the matched text from the magres file for the `isc` does not coincide with the number of atoms."
                )
                return []

            # Parse spin-spin couplings for each contribution and their refs to the specific `AtomsState`
            for i, coupled_atom_data in enumerate(data):
                for j, atom_data in enumerate(coupled_atom_data):
                    values = np.transpose(np.reshape(atom_data[4:], (3, 3)))
                    sec_isc = SpinSpinCoupling(
                        type=contribution,
                        entity_ref_1=cell.atoms_state[i],
                        entity_ref_2=cell.atoms_state[j],
                    )
                    sec_isc.reduced_value = values * 1e19 * ureg("K^2/J")
                    spin_spin_couplings.append(sec_isc)
        return spin_spin_couplings

    def parse_magnetic_susceptibilities(
        self, magres_data: TextParser, logger: "BoundLogger"
    ) -> list[MagneticSusceptibility]:
        """
        Parse the magnetic susceptibilities from the magres file.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[MagneticSusceptibility]: The list of parsed `MagneticSusceptibility` sections.
        """
        data = magres_data.get("sus", [])
        if np.size(data) != 9:
            logger.warning(
                "The shape of the matched text from the magres file for the `sus` does not coincide with 9 (3x3 tensor)."
            )
            return []
        values = np.transpose(np.reshape(data, (3, 3)))
        sec_sus = MagneticSusceptibility(scale_dimension="macroscopic")
        sec_sus.value = values * 1e-6 * ureg("dimensionless")
        return [sec_sus]

    def parse_outputs(
        self, simulation: Simulation, logger: "BoundLogger"
    ) -> Optional[Outputs]:
        """
        Parse the `Outputs` section. It extracts the information of the [magres][/magres] block and passes
        it as input for parsing the corresponding properties. It also assigns references to the `ModelMethod` and `ModelSystem`
        sections used for the simulation.

        Args:
            simulation (Simulation): The `Simulation` section used to resolve the references.
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[Outputs]: The parsed `Outputs` section.
        """
        # Initial check on `Simulation.model_system` and store the number of `AtomsState` in the
        # cell for checks of the output properties blocks
        if simulation.model_system is None:
            logger.warning(
                "Could not find the `ModelSystem` that the outputs reference to."
            )
            return None
        outputs = Outputs(
            model_method_ref=simulation.model_method[-1],
            model_system_ref=simulation.model_system[-1],
        )
        if (
            not simulation.model_system[-1].cell
            or not simulation.model_system[-1].cell[-1].atoms_state
        ):
            logger.warning(
                "Could not find the `cell` sub-section or the `AtomsState` list under it."
            )
            return None
        cell = simulation.model_system[-1].cell[-1]

        # Check if [magres][/magres] was correctly parsed
        magres_data = self.magres_file_parser.get("magres")
        if not magres_data:
            logger.warning("Could not find [magres] data block in magres file.")
            return None

        # Parse `MagneticShieldingTensor`
        ms = self.parse_magnetic_shieldings(
            magres_data=magres_data, cell=cell, logger=logger
        )
        if len(ms) > 0:
            outputs.magnetic_shieldings = ms

        # Parse `ElectricFieldGradient`
        efg = self.parse_electric_field_gradients(
            magres_data=magres_data, cell=cell, logger=logger
        )
        if (
            len(efg.efg_total) > 0
            or len(efg.efg_local) > 0
            or len(efg.efg_nonlocal) > 0
        ):
            efg.model_system_ref = simulation.model_system[-1]
            efg.model_method_ref = simulation.model_method[-1]
            outputs.electric_field_gradients.append(efg)

        # Parse `SpinSpinCoupling`
        isc = self.parse_spin_spin_couplings(
            magres_data=magres_data, cell=cell, logger=logger
        )
        if len(isc) > 0:
            outputs.spin_spin_couplings = isc

        # Parse `MagneticSusceptibility`
        mag_sus = self.parse_magnetic_susceptibilities(
            magres_data=magres_data, logger=logger
        )
        if len(mag_sus) > 0:
            outputs.magnetic_susceptibilities = mag_sus

        return outputs

    def parse_nmr_magres_file_format(
        self, nmr_first_principles_archive: "EntryArchive"
    ):
        """
        Automatically parses the NMR Magres workflow. Here, `self.archive` is the
        NMR magres archive in which we will link the original NMR first principles (CASTEP
        or QuantumESPRESSO) entry.

        Args:
            nmr_first_principles_archive (EntryArchive): the NMR (first principles) CASTEP or QuantumESPRESSO archive.
        """
        workflow = NMRMagRes(method=NMRMagResMethod(), results=NMRMagResResults())
        workflow.name = "NMR Magres"

        # ! Fix this once CASTEP and QuantumESPRESSO use the new `nomad-simulations` schema under 'data'
        # Method
        # method_nmr = extract_section(nmr_first_principles_archive, ['run', 'method'])
        # workflow.method.nmr_method_ref = method_nmr

        # Inputs and Outputs
        # ! Fix this to extract `input_structure` from `nmr_first_principles_archive` once
        # ! CASTEP and QuantumESPRESSO use the new `nomad-simulations` schema under 'data'
        input_structure = extract_section(self.archive, ["data", "model_system"])
        nmr_magres_calculation = extract_section(self.archive, ["data", "outputs"])
        if input_structure:
            workflow.m_add_sub_section(
                NMRMagRes.inputs, Link(name="Input structure", section=input_structure)
            )
        if nmr_magres_calculation:
            workflow.m_add_sub_section(
                NMRMagRes.outputs,
                Link(name="Output NMR calculation", section=nmr_magres_calculation),
            )

        # NMR (first principles) task
        # ! Fix this once CASTEP and QuantumESPRESSO use the new `nomad-simulations` schema under 'data'
        program_name = nmr_first_principles_archive.run[-1].program.name
        if nmr_first_principles_archive.workflow2:
            task = TaskReference(task=nmr_first_principles_archive.workflow2)
            task.name = f"NMR FirstPrinciples {program_name}"
            if input_structure:
                task.inputs = [Link(name="Input structure", section=input_structure)]
            if nmr_magres_calculation:
                task.outputs = [
                    Link(
                        name="Output NMR calculation",
                        section=nmr_magres_calculation,
                    )
                ]
            workflow.m_add_sub_section(NMRMagRes.tasks, task)

        self.archive.workflow2 = workflow

    def parse_json_file(self, filepath: str, logger: "BoundLogger") -> Optional[CCPNCMetadata]:
        """Parse the JSON file and extract relevant information."""
        magres_json_file = get_files(
            pattern="MRD*.json", filepath=filepath, stripname=self.basename
        )
        if not magres_json_file:
            logger.warning("No JSON file found.")
            return None
        with open(magres_json_file[0]) as f:
            magres_json_data = json.load(f)
        ccpnc_metadata = CCPNCMetadata()
        material_properties = MaterialProperties()
        orcid = ORCID()
        ccpnc_record = CCPNCRecord()
        external_database_reference = ExternalDatabaseReference()
        free_text_metadata = FreeTextMetadata()

        material_properties.chemical_name = magres_json_data.get("chemname", "")
        material_properties.formula = magres_json_data.get("formula", "")
        material_properties.stoichiometry = magres_json_data.get("stochiometry", "")
        material_properties.elements_ratios = magres_json_data.get("elements_ratios", "")
        # material_properties.chemical_name_tokens =
        orcid.orcid_id = magres_json_data.get("ORCID", "")
        # ccpnc_record.visible =
        ccpnc_record.immutable_id = magres_json_data.get("immutable_id", "")
        version_metadata = magres_json_data.get("version_metadata", {})
        external_database_reference.external_database_name = version_metadata.get("extref_type", "")
        external_database_reference.external_database_reference_code = version_metadata.get("extref_code", "")
        free_text_metadata.uploader_author_notes = version_metadata.get("notes", "")
        free_text_metadata.structural_descriptor_notes = version_metadata.get("chemform", "")

        ccpnc_metadata.material_properties = material_properties
        ccpnc_metadata.orcid = orcid
        ccpnc_metadata.ccpnc_record = ccpnc_record
        ccpnc_metadata.external_database_reference = external_database_reference
        ccpnc_metadata.free_text_metadata = free_text_metadata
        return ccpnc_metadata

    def parse(
        self,
        filepath: str,
        archive: "EntryArchive",
        logger: "BoundLogger",
        child_archives: Dict[str, EntryArchive] = None,
    ) -> None:
        self.mainfile = filepath
        self.maindir = os.path.dirname(self.mainfile)
        self.basename = os.path.basename(self.mainfile)
        self.archive = archive

        self.init_parser(logger=logger)
        self._check_units_magres(logger=logger)

        # Adding Simulation to data
        simulation = Simulation()
        calculation_params = self.magres_file_parser.get("calculation", {})
        if calculation_params.get("code", "") != "CASTEP":
            logger.error(
                "Only CASTEP-based NMR simulations are supported by the magres parser."
            )
            return
        simulation.program = Program(
            name=calculation_params.get("code", ""),
            version=calculation_params.get("code_version", ""),
        )
        archive.data = simulation

        # `ModelSystem` parsing
        model_system = self.parse_model_system(logger=logger)
        if model_system is not None:
            simulation.model_system.append(model_system)

        # `ModelMethod` parsing
        model_method = self.parse_model_method(calculation_params=calculation_params)
        simulation.model_method.append(model_method)

        # `Outputs` parsing
        outputs = self.parse_outputs(simulation=simulation, logger=logger)
        if outputs is not None:
            simulation.outputs.append(outputs)

        # Parse JSON file and extract metadata
        ccpnc_metadata = self.parse_json_file(filepath=self.mainfile, logger=logger)
        if ccpnc_metadata:
            simulation.ccpnc_metadata = ccpnc_metadata

        archive.data = simulation
        # ! this will only work after the CASTEP and QE plugin parsers are defined
        # Try to resolve the `entry_id` and `mainfile` of other entries in the upload to connect the magres entry with the CASTEP or QuantumESPRESSO entry
        filepath_stripped = self.mainfile.split("raw/")[-1]
        metadata = []
        try:
            upload_id = self.archive.metadata.upload_id
            search_ids = search(
                owner="visible",
                user_id=self.archive.metadata.main_author.user_id,
                query={"upload_id": upload_id},
                required=MetadataRequired(include=["entry_id", "mainfile"]),
            ).data
            metadata = [[sid["entry_id"], sid["mainfile"]] for sid in search_ids]
        except Exception:
            logger.warning(
                "Could not resolve the entry_id and mainfile of other entries in the upload."
            )
            return
        for entry_id, mainfile in metadata:
            if mainfile == filepath_stripped:  # we skip the current parsed mainfile
                continue
            # We try to load the archive from its context and connect both the CASTEP and the magres entries
            # ? add more checks on the system information for the connection?
            try:
                entry_archive = self.archive.m_context.load_archive(
                    entry_id, upload_id, None
                )
                # ! Fix this when CASTEP parser uses the new `data` schema
                method_label = entry_archive.run[-1].method[-1].label
                if method_label == "NMR":
                    castep_archive = entry_archive
                    # We write the workflow NMRMagRes directly in the magres entry
                    self.parse_nmr_magres_file_format(
                        nmr_first_principles_archive=castep_archive
                    )
                    break
            except Exception:
                continue

        # Populate `CCPNCMetadata` (note the `pattern` has to match the aux file generated by the MongoDB CCP-NC)
        magres_json_file = get_files(
            pattern="magres*.json", filepath=self.mainfile, stripname=self.basename
        )
        if magres_json_file is not None:
            ccpnc_metadata = CCPNCMetadata()
            # TODO: populate `ccpnc_metadata` model from `magres_json_file` HERE
            # ...
            # ...
            simulation.ccpnc_metadata = ccpnc_metadata
