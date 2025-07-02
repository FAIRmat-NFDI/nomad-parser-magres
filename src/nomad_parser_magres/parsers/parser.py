import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from collections import defaultdict

from nomad.config import config
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.workflow import Link, TaskReference
from nomad.parsing import MatchingParser
from nomad.parsing.file_parser import Quantity, TextParser
from nomad.units import ureg
from nomad.utils import extract_section
from nomad_nmr_schema.schema_packages.schema_package import (
    ElectricFieldGradient,
    IndirectSpinSpinCoupling,
    IndirectSpinSpinCouplingFermiContact,
    IndirectSpinSpinCouplingOrbitalDiamagnetic,
    IndirectSpinSpinCouplingOrbitalParamagnetic,
    IndirectSpinSpinCouplingSpinDipolar,
    MagneticShielding,
    MagneticSusceptibility,
    Outputs,
)
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import (
    DFT,
    ModelMethod,
    XCFunctional,
)
from nomad_simulations.schema_packages.model_system import AtomicCell, Cell, ModelSystem
from nomad_simulations.schema_packages.numerical_settings import KMesh, KSpace

from .workflow import (
    NMRMagRes,
    NMRMagResMethod,
    NMRMagResResults,
)

re_float = r' *[-+]?\d+\.\d*(?:[Ee][-+]\d+)? *'


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
                            'atom',
                            rf'atom *([a-zA-Z]+) *(\S+) *([\d]+) *({re_float * 3})',
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
    """
    MagresParser is a specialized parser for handling NMR Magres files.
    It extends the MatchingParser class and provides methods
    to parse various sections of a Magres file, including atomic cell,
    model system, exchange-correlation functional, magnetic shieldings,
    electric field gradients, spin-spin couplings, and magnetic susceptibilities.
    The parser also checks for unit consistency and initializes the MagresFileParser.

    Attributes:
        simulation_class: The class representing the simulation section.
        program_class: The class representing the program section.
        cell_class: The class representing the cell section.
        model_system_class: The class representing the model system section.
        model_method_class: The class representing the model method section.
        atom_state_class: The class representing the atom state section.
        magres_outputs_class: The class representing the outputs section.
        indirect_spin_spin_couplings_class: The class representing the spin-spin coupling section.
        indirect_spin_spin_couplings_fc_class: The class representing the Fermi contact spin-spin coupling section.
        indirect_spin_spin_couplings_orbital_d_class: The class representing the orbital diamagnetic spin-spin coupling section.
        indirect_spin_spin_couplings_orbital_p_class: The class representing the orbital paramagnetic spin-spin coupling section.
        indirect_spin_spin_couplings_spin_class: The class representing the spin dipolar spin-spin coupling section.
        e_field_gradient_class: The class representing the electric field gradient section.
        mag_susceptibility_class: The class representing the magnetic susceptibility section.
        mag_shielding: The class representing the magnetic shielding tensor section.
        workflow_class: The class representing the workflow section.
        workflow_method_class: The class representing the workflow method section.
        workflow_results_class: The class representing the workflow results section.

    Methods:
        __init__(*args, **kwargs): Initializes the MagresParser with optional arguments.
        _check_units_magres(logger: "BoundLogger") -> None:
            Checks if the units of the NMR quantities are magres standard.
        init_parser(logger: "BoundLogger") -> None:
            Initializes the MagresFileParser with the mainfile and logger.
        parse_atomic_cell(atoms: Optional[TextParser],
            logger: "BoundLogger") -> Optional[AtomicCell]:
            Parses the AtomicCell section from the magres file.
        parse_model_system(logger: "BoundLogger") -> Optional["MagresParser.model_system_class"]:
            Parses the model system section from the magres file.
        parse_xc_functional(calculation_params: Optional[TextParser]) -> list[XCFunctional]:
            Parses the exchange-correlation functional information from the magres file.
        parse_model_method(
            calculation_params: Optional[TextParser]
            ) -> "MagresParser.model_method_class":
            Parses the model method section by extracting information about the NMR method.
        parse_magnetic_shieldings(magres_data: TextParser,
            cell: "MagresParser.cell_class",
            logger: "BoundLogger"
            ) -> list["MagresParser.mag_shielding_tensor"]:
            Parses the magnetic shieldings from the magres file.
        parse_electric_field_gradients(magres_data: TextParser,
            cell: "MagresParser.cell_class",
            logger: "BoundLogger"
            ) -> "MagresParser.e_field_gradients_class":
            Parses the electric field gradients from the magres file.
        parse_spin_spin_couplings(magres_data: TextParser,
            cell: "MagresParser.cell_class",
            logger: "BoundLogger") -> list["MagresParser.spin_spin_couplings_class"]:
            Parses the spin-spin couplings from the magres file.
        parse_magnetic_susceptibilities(magres_data: TextParser,
            logger: "BoundLogger") -> list["MagresParser.mag_susceptibility_class"]:
            Parses the magnetic susceptibilities from the magres file.
        parse_outputs(simulation: "MagresParser.simulation_class",
            logger: "BoundLogger") -> Optional["MagresParser.magres_outputs_class"]:
            Parses the outputs section and assigns references
            to the model method and model system sections.
        parse_nmr_magres_file_format(nmr_first_principles_archive: "EntryArchive"):
            Automatically parses the NMR Magres workflow
            and links the original NMR first principles entry.
        parse(filepath: str, archive: "EntryArchive",
            logger: "BoundLogger", child_archives: dict[str, EntryArchive] = None) -> None:
            Parses the magres file and populates the archive with the parsed data.
    """

    # Be careful when changing this class references
    # as you might incur in AttributeError exceptions or other issues
    #
    # Data section classes:
    simulation_class = Simulation
    program_class = Program
    cell_class = Cell
    model_system_class = ModelSystem
    model_method_class = ModelMethod
    atom_state_class = AtomsState
    magres_outputs_class = Outputs
    indirect_spin_spin_couplings_class = IndirectSpinSpinCoupling
    indirect_spin_spin_couplings_fc_class = IndirectSpinSpinCouplingFermiContact
    indirect_spin_spin_couplings_orbital_d_class = (
        IndirectSpinSpinCouplingOrbitalDiamagnetic
    )
    indirect_spin_spin_couplings_orbital_p_class = (
        IndirectSpinSpinCouplingOrbitalParamagnetic
    )
    indirect_spin_spin_couplings_spin_class = IndirectSpinSpinCouplingSpinDipolar
    e_field_gradient_class = ElectricFieldGradient
    mag_susceptibility_class = MagneticSusceptibility
    mag_shielding = MagneticShielding
    # Worḱflow section classes:
    workflow_class = NMRMagRes
    workflow_method_class = NMRMagResMethod
    workflow_results_class = NMRMagResResults

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
        self, atoms: TextParser | None, logger: 'BoundLogger'
    ) -> AtomicCell | None:
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
            logger.warning('Could not find atomic structure in magres file.')
            return None
        atomic_cell = AtomicCell()

        # Parse `lattice_vectors` and `periodic_boundary_conditions`
        try:
            lattice_vectors = np.reshape(np.array(atoms.get('lattice', [])), (3, 3))
            atomic_cell.lattice_vectors = lattice_vectors * ureg.angstrom
            pbc = (
                [True, True, True]
                if lattice_vectors is not None
                else [False, False, False]
            )
            atomic_cell.periodic_boundary_conditions = pbc
        except Exception:
            logger.warning(
                'Could not parse `lattice_vectors` and `periodic_boundary_conditions`.'
            )
            return None

        return atomic_cell

    def parse_model_system(
        self, logger: 'BoundLogger'
    ) -> Optional['MagresParser.model_system_class']:
        """
        Parse the `MagresParser.model_system_class` section from the magres file if the [atoms][/atoms] section
        in the magres file was correctly matched.

        Args:
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[MagresParser.model_system_class]: The parsed `MagresParser.model_system_class` section.
        """
        # Check if [atoms][/atoms] was correctly parsed
        atoms = self.magres_file_parser.get('atoms')
        if not atoms:
            logger.warning('Could not find atomic structure in magres file.')
            return None

        # Parse `MagresParser.model_system_class` and its `cell`
        model_system = self.model_system_class()
        model_system.is_representative = True
        atomic_cell = self.parse_atomic_cell(atoms=atoms, logger=logger)
        model_system.cell.append(atomic_cell)

        # Parse `positions` and `MagresParser.atom_state_class` list
        atoms_list = atoms.get('atom', [])
        if len(atoms_list) == 0:
            logger.warning(
                'Could not find atom `positions` and their chemical symbols in magres file.'
            )
            return None
        positions = []
        particle_states = []

        for atom in atoms_list:
            particle_states.append(
                self.atom_state_class(chemical_symbol=atom[0], label=atom[1])
            )
            positions.append(atom[3:])
        model_system.positions = positions * ureg.angstrom
        model_system.particle_states = particle_states

        self.build_particle_lookup(model_system, logger)
        self.build_particle_pair_lookup(model_system, logger)

        return model_system

    def parse_xc_functional(
        self, calculation_params: TextParser | None
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
        self, calculation_params: TextParser | None
    ) -> "MagresParser.model_method_class":
        """
        Parse the `MagresParser.model_method_class` section by extracting information about the NMR method: basis set,
        exchange-correlation functional, cutoff energy, and K mesh.

        Note: only CASTEP-like method parameters are currently being supported.

        Args:
            calculation_params (Optional[TextParser]): The parsed [calculation][/calculation] block parameters.

        Returns:
            Optional[MagresParser.model_method_class]: The parsed `MagresParser.model_method_class` section.
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
        self,
        magres_data: TextParser,
        cell: "MagresParser.cell_class",
        logger: "BoundLogger",
    ) -> list["MagresParser.mag_shielding_tensor"]:
        """
        Parse the magnetic shieldings from the magres file and assign `entity_ref` to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[MagresParser.mag_shielding_tensor]: The list of parsed `MagresParser.mag_shielding_tensor` sections.
        """
        n_atoms = len(cell.atoms_state)
        data = magres_data.get("ms", [])

        # Initial check on the size of the matched text
        if np.size(data) != n_atoms * (9 + 2):  # 2 extra columns with atom labels
            logger.warning(
                "The shape of the matched text from the magres file for the `ms` does not coincide with the number of atoms."
            )
            return []

        # Parse magnetic shieldings and their refs to the specific `MagresParser.atom_state_class`
        magnetic_shieldings = []
        for i, atom_data in enumerate(data):
            # values = np.transpose(np.reshape(atom_data[2:], (3, 3)))
            values = np.transpose(np.reshape(atom_data[2:], (3, 3)))
            sec_ms = self.mag_shielding_tensor(entity_ref=cell.atoms_state[i])
            sec_ms.value = values * 1e-6 * ureg("dimensionless")
            magnetic_shieldings.append(sec_ms)
        return magnetic_shieldings

    def parse_electric_field_gradients(
        self,
        magres_data: TextParser,
        cell: "MagresParser.cell_class",
        logger: "BoundLogger",
    ) -> "MagresParser.e_field_gradients_class":
        """
        Parse the electric field gradients from the magres file and assign `entity_ref` to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            self.: The parsed `self.e_field_gradients_class` section.
        """
        n_atoms = len(cell.atoms_state)
        efg_contributions = {
            "efg_local": "local",
            "efg_nonlocal": "non_local",
            "efg": "total",
        }
        # electric_field_gradients = []
        electric_field_gradients = self.e_field_gradients_class()
        for tag, contribution in efg_contributions.items():
            data = magres_data.get(tag, [])

            # Initial check on the size of the matched text
            if np.size(data) != n_atoms * (9 + 2):  # 2 extra columns with atom labels
                logger.warning(
                    "The shape of the matched text from the magres file for the `efg` does not coincide with the number of atoms."
                )
                # return []
                continue  # Log a warning and continue processing the remaining tags

            # Parse electronic field gradients for each contribution and their refs to the specific `MagresParser.atom_state_class`
            for i, atom_data in enumerate(data):
                # values = np.transpose(np.reshape(atom_data[2:], (3, 3)))
                values = np.reshape(atom_data[2:], (3, 3))  # no need to transpose
                sec_efg = self.e_field_gradient_class(
                    type=contribution, entity_ref=cell.atoms_state[i]
                )
                sec_efg.value = np.transpose(values) * 9.717362e21 * ureg("V/m^2")
                # electric_field_gradients.append(sec_efg)
                if contribution == "total":
                    electric_field_gradients.efg_total.append(sec_efg)
                elif contribution == "local":
                    electric_field_gradients.efg_local.append(sec_efg)
                elif contribution == "non_local":
                    electric_field_gradients.efg_nonlocal.append(sec_efg)
        return electric_field_gradients

    def parse_spin_spin_couplings(
        self,
        magres_data: TextParser,
        cell: "MagresParser.cell_class",
        logger: "BoundLogger",
    ) -> list["MagresParser.spin_spin_couplings_class"]:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[self.spin_spin_couplings_class]: The list of parsed `self.spin_spin_couplings_class` sections.
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

            # Parse spin-spin couplings for each contribution and their refs to the specific `MagresParser.atom_state_class`
            for i, coupled_atom_data in enumerate(data):
                for j, atom_data in enumerate(coupled_atom_data):
                    values = np.transpose(np.reshape(atom_data[4:], (3, 3)))
                    sec_isc = self.spin_spin_couplings_class(
                        type=contribution,
                        entity_ref_1=cell.atoms_state[i],
                        entity_ref_2=cell.atoms_state[j],
                    )
                    sec_isc.reduced_value = values * 1e19 * ureg("K^2/J")
                    spin_spin_couplings.append(sec_isc)
        return spin_spin_couplings

    def parse_magnetic_susceptibilities(
        self, magres_data: TextParser, logger: "BoundLogger"
    ) -> list["MagresParser.mag_susceptibility_class"]:
        """
        Parse the magnetic susceptibilities from the magres file.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[MagresParser.mag_susceptibility_class]: The list of parsed `MagresParser.mag_susceptibility_class` sections.
        """
        data = magres_data.get("sus", [])
        if np.size(data) != 9:
            logger.warning(
                "The shape of the matched text from the magres file for the `sus` does not coincide with 9 (3x3 tensor)."
            )
            return []
        values = np.transpose(np.reshape(data, (3, 3)))
        sec_sus = self.mag_susceptibility_class(scale_dimension="macroscopic")
        sec_sus.value = values * 1e-6 * ureg("dimensionless")
        return [sec_sus]

    def parse_outputs(
        self, simulation: "MagresParser.simulation_class", logger: "BoundLogger"
    ) -> Optional["MagresParser.magres_outputs_class"]:
        """
        Parse the `self.magres_outputs_class` section. It extracts the information of the [magres][/magres] block and passes
        it as input for parsing the corresponding properties. It also assigns references to the `MagresParser.model_method_class` and `MagresParser.model_system_class`
        sections used for the simulation.

        Args:
            simulation ('MagresParser.simulation_class'): The `MagresParser.simulation_class` section used to resolve the references.
            logger (BoundLogger): The logger to log messages.

        Returns:
            Optional[self.magres_outputs_class]: The parsed `self.magres_outputs_class` section.
        """
        # Initial check on `MagresParser.simulation_class.model_system` and store the number of `MagresParser.atom_state_class` in the
        # cell for checks of the output properties blocks
        if simulation.model_system is None:
            logger.warning(
                "Could not find the `MagresParser.model_system_class` that the outputs reference to."
            )
            return None
        outputs = self.magres_outputs_class(
            model_method_ref=simulation.model_method[-1],
            model_system_ref=simulation.model_system[-1],
        )
        if (
            not simulation.model_system[-1].cell
            or not simulation.model_system[-1].cell[-1].atoms_state
        ):
            logger.warning(
                "Could not find the `cell` sub-section or the `MagresParser.atom_state_class` list under it."
            )
            return None
        cell = simulation.model_system[-1].cell[-1]

        # Check if [magres][/magres] was correctly parsed
        magres_data = self.magres_file_parser.get("magres")
        if not magres_data:
            logger.warning("Could not find [magres] data block in magres file.")
            return None

        # Parse `MagresParser.mag_shielding_tensor`
        ms = self.parse_magnetic_shieldings(
            magres_data=magres_data, cell=cell, logger=logger
        )
        if len(ms) > 0:
            outputs.magnetic_shieldings = ms

        # Parse `MagresParser.e_field_gradient_class`
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

        # Parse `self.spin_spin_couplings_class`
        isc = self.parse_spin_spin_couplings(
            magres_data=magres_data, cell=cell, logger=logger
        )
        if len(isc) > 0:
            outputs.spin_spin_couplings = isc

        # Parse `MagresParser.mag_susceptibility_class`
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
        workflow = self.workflow_class(
            method=self.workflow_method_class(), results=self.workflow_results_class()
        )
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
                self.workflow_class.inputs,
                Link(name="Input structure", section=input_structure),
            )
        if nmr_magres_calculation:
            workflow.m_add_sub_section(
                self.workflow_class.outputs,
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
            workflow.m_add_sub_section(self.workflow_class.tasks, task)

        self.archive.workflow2 = workflow

    def parse(
        self,
        filepath: str,
        archive: "EntryArchive",
        logger: "BoundLogger",
        child_archives: dict[str, EntryArchive] = None,
    ) -> None:
        self.mainfile = filepath
        self.maindir = os.path.dirname(self.mainfile)
        self.basename = os.path.basename(self.mainfile)
        self.archive = archive

        self.init_parser(logger=logger)
        self._check_units_magres(logger=logger)

        # Adding self.simulation_class to data
        simulation = self.simulation_class()
        calculation_params = self.magres_file_parser.get("calculation", {})
        if calculation_params.get("code", "") != "CASTEP":
            logger.error(
                "Only CASTEP-based NMR simulations are supported by the magres parser."
            )
            return
        simulation.program = self.program_class(
            name=calculation_params.get("code", ""),
            version=calculation_params.get("code_version", ""),
        )
        archive.data = simulation

        # `MagresParser.model_system_class` parsing
        model_system = self.parse_model_system(logger=logger)
        if model_system is not None:
            simulation.model_system.append(model_system)

        # `MagresParser.model_method_class` parsing
        model_method = self.parse_model_method(calculation_params=calculation_params)
        simulation.model_method.append(model_method)

        # `self.magres_outputs` parsing
        outputs = self.parse_outputs(simulation=simulation, logger=logger)
        if outputs is not None:
            simulation.outputs.append(outputs)

        archive.data = simulation
        # ! this will only work after the CASTEP and QE plugin parsers are defined
        # Try to resolve the `entry_id` and `mainfile` of other entries in the upload to connect the magres entry with the CASTEP or QuantumESPRESSO entry
        filepath_stripped = self.mainfile.split("raw/")[-1]
        metadata = []
        try:
            from nomad.app.v1.models.models import MetadataRequired
            from nomad.search import search
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
            #if mainfile == filepath_stripped:  # we skip the current parsed mainfile
            #    continue
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
                    # We write the workflow MagresParser.workflow_class directly in the magres entry
                    self.parse_nmr_magres_file_format(
                        nmr_first_principles_archive=castep_archive
                    )
                    break
            except Exception:
                continue
