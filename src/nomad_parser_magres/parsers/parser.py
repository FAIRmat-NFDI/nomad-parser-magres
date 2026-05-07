import os
import re
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
from nomad_simulations.schema_packages.model_system import ModelSystem, Representation
from nomad_simulations.schema_packages.numerical_settings import KMesh, KSpace

from .workflow import (
    NMRMagRes,
    NMRMagResMethod,
    NMRMagResResults,
)

# Updated regex to match floating point numbers from various scientific notations
re_float = r' *[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[Ee][-+]?\d+)? *'


class MagresFileParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        self._quantities = [
            Quantity('lattice_units', r'units *lattice *([a-zA-Z]+)'),
            Quantity('atom_units', r'units *atom *([a-zA-Z]+)'),
            Quantity('ms_units', r'units *ms *([a-zA-Z]+)'),
            Quantity('efg_units', r'units *efg *([a-zA-Z]+)'),
            Quantity('isc_units', r'units *isc *([a-zA-Z\^\d\.\-]+)'),
            Quantity('isc_fc_units', r'units *isc_fc *([a-zA-Z\^\d\.\-]+)'),
            Quantity('isc_spin_units', r'units *isc_spin *([a-zA-Z\^\d\.\-]+)'),
            Quantity(
                'isc_orbital_p_units', r'units *isc_orbital_p *([a-zA-Z\^\d\.\-]+)'
            ),
            Quantity(
                'isc_orbital_d_units', r'units *isc_orbital_d *([a-zA-Z\^\d\.\-]+)'
            ),
            Quantity('sus_units', r'units *sus *([a-zA-Z\^\d\.\-]+)'),
            Quantity('cutoffenergy_units', r'units *calc\_cutoffenergy *([a-zA-Z]+)'),
            Quantity(
                'calculation',
                r'([\[\<]*calculation[\>\]]*[\s\S]+?)(?:[\[\<]*\/calculation[\>\]]*)',
                sub_parser=TextParser(
                    quantities=[
                        # Quantity('code', r'calc\_code *([a-zA-Z]+)'),
                        Quantity('code', r'calc\_code\s+([^\n]+)'),
                        Quantity(
                            'code_version', r'calc\_code\_version *([a-zA-Z\d\.]+)'
                        ),
                        Quantity(
                            'code_hgversion',
                            r'calc\_code\_hgversion ([a-zA-Z\d\:\+\s]*)\n',
                            flatten=False,
                        ),
                        Quantity(
                            'code_platform', r'calc\_code\_platform *([a-zA-Z\d\_]+)'
                        ),
                        Quantity('name', r'calc\_name *([\w]+)'),
                        Quantity('comment', r'calc\_comment *([\w]+)'),
                        Quantity('xcfunctional', r'calc\_xcfunctional *([\w]+)'),
                        Quantity(
                            'cutoffenergy',
                            rf'calc\_cutoffenergy({re_float})(?P<__unit>\w+)',
                        ),
                        Quantity(
                            'pspot',
                            r'calc\_pspot *([\w]+) *([\w\.\|\(\)\=\:]+)',
                            repeats=True,
                        ),
                        Quantity(
                            'kpoint_mp_grid',
                            r'calc\_kpoint\_mp\_grid *([\w]+) *([\w]+) *([\w]+)',
                        ),
                        Quantity(
                            'kpoint_mp_offset',
                            r'calc_kpoint_mp_offset\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)',
                        ),
                    ]
                ),
            ),
            Quantity(
                'atoms',
                r'([\[\<]*atoms[\>\]]*[\s\S]+?)(?:[\[\<]*\/atoms[\>\]]*)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity('lattice', rf'lattice({re_float * 9})'),
                        Quantity('symmetry', r'symmetry *([\w\-\+\,]+)', repeats=True),
                        Quantity(
                            'atom',
                            rf'atom *([a-zA-Z]+) *(\S+) *([\d]+) *({re_float * 3})',
                            repeats=True,
                        ),
                    ]
                ),
            ),
            Quantity(
                'magres',
                r'([\[\<]*magres[\>\]]*[\s\S]+?)(?:[\[\<]*\/magres[\>\]]*)',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'ms', rf'ms *(\w+) *(\d+)({re_float * 9})', repeats=True
                        ),
                        Quantity(
                            'efg', rf'efg *(\w+) *(\d+)({re_float * 9})', repeats=True
                        ),
                        Quantity(
                            'isc',
                            rf'isc *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})',
                            repeats=True,
                        ),
                        Quantity(
                            'isc_fc',
                            rf'isc_fc *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})',
                            repeats=True,
                        ),
                        Quantity(
                            'isc_orbital_p',
                            rf'isc_orbital_p *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})',
                            repeats=True,
                        ),
                        Quantity(
                            'isc_orbital_d',
                            rf'isc_orbital_d *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})',
                            repeats=True,
                        ),
                        Quantity(
                            'isc_spin',
                            rf'isc_spin *(\w+) *(\d+) *(\w+) *(\d+)({re_float * 9})',
                            repeats=True,
                        ),
                        Quantity('sus', rf'sus *({re_float * 9})', repeats=True),
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
        parse_model_system(logger: "BoundLogger") -> Optional["MagresParser.model_system_class"]:
            Parses the model system section from the magres file.
        parse_xc_functional(calculation_params: Optional[TextParser]) -> list[XCFunctional]:
            Parses the exchange-correlation functional information from the magres file.
        parse_model_method(
            calculation_params: Optional[TextParser]
            ) -> "MagresParser.model_method_class":
            Parses the model method section by extracting information about the NMR method.
        parse_magnetic_shieldings(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger"
            ) -> list["MagresParser.mag_shielding"]:
            Parses the magnetic shieldings from the magres file.
        parse_electric_field_gradients(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger"
            ) -> list["MagresParser.e_field_gradient_class"]:
            Parses the electric field gradients from the magres file.
        parse_indirect_spin_spin_couplings(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger") -> list["MagresParser.indirect_spin_spin_couplings_class"]:
            Parses the indirect spin-spin couplings (total contribution) from the magres file.
        parse_indirect_spin_spin_couplings_fc(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger") -> list["MagresParser.indirect_spin_spin_couplings_fc_class"]:
            Parses the Fermi contact contribution to the indirect spin-spin couplings from the magres file.
        parse_indirect_spin_spin_couplings_orbital_d(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger") -> list["MagresParser.indirect_spin_spin_couplings_orbital_d_class"]:
            Parses the orbital diamagnetic contribution to the indirect spin-spin couplings from the magres file.
        parse_indirect_spin_spin_couplings_orbital_p(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger") -> list["MagresParser.indirect_spin_spin_couplings_orbital_p_class"]:
            Parses the orbital paramagnetic contribution to the indirect spin-spin couplings from the magres file.
        parse_indirect_spin_spin_couplings_spin(magres_data: TextParser,
            atom_state_class: "MagresParser.atom_state_class",
            model_system: "MagresParser.model_system_class",
            logger: "BoundLogger") -> list["MagresParser.indirect_spin_spin_couplings_spin_class"]:
            Parses the spin dipolar contribution to the indirect spin-spin couplings from the magres file.
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
    # Workflow section classes:
    workflow_class = NMRMagRes
    workflow_method_class = NMRMagResMethod
    workflow_results_class = NMRMagResResults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magres_file_parser = MagresFileParser()
        # Jacob's ladder classification mapping
        self._xc_functional_type_map = {
            'LDA': 'LDA',
            'PBE': 'GGA',
            'PW91': 'GGA',
            'RPBE': 'GGA',
            'WC': 'GGA',
            'PBESOL': 'GGA',
            'BLYP': 'GGA',
            'B3LYP': 'hyb_GGA',
            'PBE0': 'hyb_GGA',
            'HSE03': 'hyb_GGA',
            'HSE06': 'hyb_GGA',
            'RSCAN': 'meta_GGA',
            'HF': 'HF'
        }
        self.particle_lookup = {}  # To ensure associaton of particle states with correct magres data
        self.particle_pair_lookup = {}  # To ensure association of particle pairs with correct magres data

    def _check_units_magres(self, logger: 'BoundLogger') -> None:
        """
        Check if the units of the NMR quantities are magres standard. If not, a warning
        is issued and the default units are used.
        """
        allowed_units = {
            'lattice': 'Angstrom',
            'atom': 'Angstrom',
            'ms': 'ppm',
            'efg': 'au',
            'isc': '10^19.T^2.J^-1',
            'isc_fc': '10^19.T^2.J^-1',
            'isc_orbital_p': '10^19.T^2.J^-1',
            'isc_orbital_d': '10^19.T^2.J^-1',
            'isc_spin': '10^19.T^2.J^-1',
            'sus': '10^-6.cm^3.mol^-1',
        }
        for key, value in allowed_units.items():
            data = self.magres_file_parser.get(f'{key}_units', '')
            if data and data != value:
                logger.warning(
                    'The units of the NMR quantities are not parsed if they are not magres standard. '
                    'We will use the default units.',
                    data={
                        'quantities': key,
                        'standard_units': value,
                        'parsed_units': data,
                    },
                )

    def init_parser(self, logger: 'BoundLogger') -> None:
        """
        Initialize the `MagresFileParser` with the mainfile and logger.

        Args:
            logger (BoundLogger): The logger to log messages.
        """
        self.magres_file_parser.mainfile = self.mainfile
        self.magres_file_parser.logger = logger

    def _is_valid_version(self, version_str: str) -> bool:
        """
        Determine if calc_code_version contains meaningful version information.
        Returns False for version control artifacts, vague descriptors, or empty values.

        Args:
            version_str (str): The version string to validate

        Returns:
            bool: True if version appears to be meaningful, False otherwise
        """
        if not version_str or not version_str.strip():
            return False

        version_lower = version_str.lower().strip()

        # Check for common vague/invalid patterns
        invalid_patterns = [
            r'^(git|svn|cvs|hg|bzr)',  # Version control systems
            r'^(unknown|n/?a|none|unspecified)$',  # Vague descriptors
            r'^svn\d+$',  # SVN revision numbers like 'svn11423'
            r'^git[a-f0-9]*$',  # Git hashes or 'git' prefix
        ]

        for pattern in invalid_patterns:
            if re.match(pattern, version_lower):
                return False

        return True

    def _parse_program_info(
        self, calculation_params: dict, logger: 'BoundLogger'
        ) -> tuple[str, str]:
        """
        Parse program name and version from calculation parameters.

        Handles different formats:
        - CASTEP: calc_code='CASTEP', calc_code_version='<version>'
        - QE-GIPAW: calc_code='QE-GIPAW <version>' or 'QE-GIPAW',
                calc_code_version='<version>' or invalid values

        Returns:
            tuple: (program_name, program_version)

        Note:
            For QE-GIPAW files, if calc_code_version contains version control artifacts
            (git, svn), vague descriptors (unknown), or revision numbers (svn11423),
            the version is extracted from calc_code field instead.
            This handles legacy files with incomplete metadata gracefully.
        """
        code = calculation_params.get('code', '')
        code_version = calculation_params.get('code_version', '')

        # Handle CASTEP format (standard)
        if code == 'CASTEP':
            return 'CASTEP', code_version

        # Handle QE-GIPAW format
        # Future QE files might have properly separated fields, but older files
        # have version embedded in calc_code field
        if 'QE' in code:
            program_name = 'Quantum ESPRESSO'

            # Check if version is in the code field (e.g., "QE-GIPAW 6.x")
            if ' ' in code:
                code_parts = code.split()
                # Extract version from code field if available
                version_candidate = code_parts[-1] if len(code_parts) > 1 else ''
                # Use code_version if it's valid, otherwise fall back to version from calc_code
                if self._is_valid_version(code_version):
                    program_version = code_version
                elif version_candidate:
                    program_version = version_candidate
                    logger.warning(
                        'Using version from calc_code field due to invalid calc_code_version',
                        calc_code_version=code_version,
                        extracted_version=version_candidate,
                    )
                else:
                    program_version = 'unknown'
                    logger.error(
                        'Could not determine QE-GIPAW version from calc_code or calc_code_version.',
                        calc_code=code,
                        calc_code_version=code_version,
                    )
            else:
                # calc_code is just "QE-GIPAW" or "QE"
                if self._is_valid_version(code_version):
                    program_version = code_version
                else:
                    program_version = 'unknown'
                    logger.error(
                        'QE-GIPAW version not properly specified in magres file.',
                        calc_code_version=code_version,
                    )

            return program_name, program_version

        # Unrecognized format
        return code, code_version

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

        # Parse `MagresParser.model_system_class`
        model_system = self.model_system_class()
        model_system.is_representative = True

        # Parse lattice vectors and PBC directly on ModelSystem
        try:
            lattice_vectors = np.reshape(np.array(atoms.get('lattice', [])), (3, 3))
            model_system.lattice_vectors = lattice_vectors * ureg.angstrom
            pbc = [True, True, True] if lattice_vectors is not None else [False, False, False]
            model_system.periodic_boundary_conditions = pbc
        except Exception:
            logger.warning('Could not parse lattice_vectors and periodic_boundary_conditions.')
            return None
        
        # Parse `positions` and `MagresParser.atom_state_class` list
        atoms_list = atoms.get('atom', [])
        if len(atoms_list) == 0:
            logger.warning(
                'Could not find atom `positions` and their chemical symbols in magres file.'
            )
            return None
        positions = []
        particle_states = []
        indices = [] # for local use, to build lookup tables correctly

        for atom in atoms_list:
            label_index = f"{atom[1]}_{atom[2]}"
            particle_states.append(
                self.atom_state_class(chemical_symbol=atom[0], label=label_index)
            )
            indices.append(int(atom[2]))
            positions.append(atom[3:])
        model_system.positions = positions * ureg.angstrom
        model_system.particle_states = particle_states

        self.build_particle_lookup(model_system, indices, logger)
        self.build_particle_pair_lookup(model_system, indices, logger)

        return model_system

    def parse_xc_functional(
        self, calculation_params: TextParser | None
    ) -> Optional[XCFunctional]:
        """
        Parse the exchange-correlation functional information from the magres file.
        Creates a single XCFunctional object using high-level functional names.
        The new architecture automatically expands to components during normalization.

        Args:
            calculation_params (Optional[TextParser]): The parsed [calculation][/calculation] block parameters.

        Returns:
            Optional[XCFunctional]: The parsed `XCFunctional` section, or None if not available.
        """
        if calculation_params is None:
            return None

        xc_functional_name = calculation_params.get('xcfunctional', 'LDA')

        # Create single XCFunctional with high-level name (not libxc components)
        # Check if the normalization process automaticallys create components
        functional = XCFunctional(functional_key=xc_functional_name)
    
        return functional

    def parse_model_method(
        self, calculation_params: TextParser | None, logger=None
    ) -> 'MagresParser.model_method_class':
        """
        Parse the `MagresParser.model_method_class` section by extracting information about the NMR method: basis set,
        exchange-correlation functional, cutoff energy, and K mesh.

        Note: Only CASTEP and QE-GIPAW method parameters are currently being supported.
        QE-GIPAW generated magres files may have incomplete metadata in the calculation block, but the available data
        is parsed as-is.

        Args:
            calculation_params (Optional[TextParser]): The parsed [calculation][/calculation] block parameters.

        Returns:
            Optional[MagresParser.model_method_class]: The parsed `MagresParser.model_method_class` section.
        """
        model_method = DFT(name='NMR')

        # Parse `XCFunctinals` information
        xc_functionals = self.parse_xc_functional(calculation_params=calculation_params)
        if xc_functionals is not None:
            model_method.xc = xc_functionals
        if calculation_params is not None:
            try:
                program_name, program_version = self._parse_program_info(calculation_params, logger)
                # Note: Program info typically stored in separate Program section in simulation,
                # but we can store the method-relevant info in DFT.name if needed
                if program_name and program_name.strip():
                    model_method.name = f'NMR ({program_name})'
            except Exception as e:
                if logger:
                    logger.warning(f'Failed to parse program information: {e}')
                model_method.name = 'NMR'

        # Parse `KSpace` as a `NumericalSettings` section
        kpoint_mp_offset = calculation_params.get('kpoint_mp_offset', [0, 0, 0])
        kpoint_mp_offset = [float(x) for x in kpoint_mp_offset]
        k_mesh = KMesh(
            grid=calculation_params.get('kpoint_mp_grid', [1, 1, 1]),
            offset=kpoint_mp_offset,
        )
        model_method.numerical_settings.append(KSpace(k_mesh=[k_mesh]))

        return model_method

    def build_particle_lookup(self, model_system, indices, logger: 'BoundLogger') -> None:
        """
        Build a lookup table for particle_states using (label, index) as key,
        where index is the 1-based index among atoms with the same label.
        """

        particle_lookup = {}

        particle_states = model_system.particle_states

        # If particle_states is non-empty, create a list of indices
        # If particle_states is empty, log an error and return an empty lookup
        if not particle_states or len(particle_states) == 0:
            logger.error(
                'No particle states recorded in the model system. Cannot build lookup table.'
            )
            self.particle_lookup = {}
            return

        for idx in range(len(indices)):
            ps = particle_states[idx]
            index = indices[idx]
            ps._site_index = index  # store for reference
            label = getattr(ps, 'label', None)  # label is now label_index (e.g., H1_1)
            if label is None:
                logger.error(
                    f'`AtomsState` for particle state {ps} is missing a valid `label` attribute.'
                )
                continue
            if index is None:
                logger.error(
                    f'`AtomsState` for particle state {ps} is missing a valid `index` attribute.'
                )
                continue
            # Use label_index as the only key (label is already unique)
            particle_lookup[label] = ps

        self.particle_lookup = particle_lookup

    def build_particle_pair_lookup(self, model_system, indices, logger: 'BoundLogger') -> None:
        """
        Build a lookup table for all pairs of particle_states using ((label1, idx1),
        (label2, idx2)) as key.
        """

        label_index_to_ps = {}
        label_index_to_idx = {}

        particle_states = model_system.particle_states

        # If particle_states is non-empty, create a list of indices
        # If particle_states is empty, log an error and return an empty lookup
        if not particle_states or len(particle_states) == 0:
            logger.error(
                'No particle states recorded in the model system. Cannot build lookup table.'
            )
            self.particle_pair_lookup = {}
            return

        # Build single lookup for label/index to AtomsState and index
        for idx in range(len(indices)):
            ps = particle_states[idx]
            index = indices[idx]
            ps._site_index = index  # store for reference
            label = getattr(ps, 'label', None)  # label is now label_index (e.g., H1_1)
            if label is None:
                logger.error(
                    f'`AtomsState` for particle state {ps} is missing a valid `label` attribute.'
                )
                continue
            if idx is None:
                logger.error(
                    f'`AtomsState` for particle state {ps} is missing a valid `index` attribute.'
                )
                continue
            label_index_to_ps[label] = ps
            label_index_to_idx[label] = idx

        # Build pair lookup
        pair_lookup = {}
        for label1, ps1 in label_index_to_ps.items():
            i = label_index_to_idx[label1]
            for label2, ps2 in label_index_to_ps.items():
                j = label_index_to_idx[label2]
                pair_lookup[(label1, label2)] = (ps1, ps2, i, j)

        self.particle_pair_lookup = pair_lookup

    def extract_particle_state_and_tensor(self, atom_data, logger):
        """
        Helper function to extract the particle state and tensor values from a magres data line.
        This function is used by parse_magnetic_shieldings and parse_electric_field_gradients

        Args:
            atom_data (list): The data line for a single atom, typically containing
                [label, index, ...tensor values...].
            logger (BoundLogger): The logger to log messages.

        Returns:
            tuple: (ps, values)
                ps: The AtomsState object from self.particle_lookup corresponding to (label, index),
                    or None if not found.
                values: The 3x3 numpy array of tensor values, or None if ps is None.
        """
        label = f"{atom_data[0]}_{atom_data[1]}"
        ps = self.particle_lookup.get(label)
        if ps is None:
            logger.warning(f'Could not find atom for label_index {label}')
            return None, None
        values = np.reshape(atom_data[2:], (3, 3))
        return ps, values

    def extract_particle_pair_and_tensor(self, atom_data, logger):
        """
        Helper function to extract the particle pair and tensor values from a magres data line
        for spin-spin coupling contributions. This function is used by all indirect spin-spin
        coupling parsing functions.

        Args:
            atom_data (list): The data line for a pair of atoms, typically containing
                [label1, idx1, label2, idx2, ...tensor values...].
            logger (BoundLogger): The logger to log messages.

        Returns:
            tuple: (pair, values)
                pair: The tuple (ps1, ps2, i, j) from self.particle_pair_lookup corresponding to
                    ((label1, idx1), (label2, idx2)), or None if not found.
                values: The 3x3 numpy array of tensor values, or None if pair is None.
        """
        label1 = f"{atom_data[0]}_{atom_data[1]}"
        label2 = f"{atom_data[2]}_{atom_data[3]}"
        values = np.reshape(atom_data[4:], (3, 3))

        pair = self.particle_pair_lookup.get((label1, label2))
        if pair is None:
            logger.warning(
                f'Could not find AtomsState pair for {label1}-{label2}'
            )
            return None, None
        return pair, values

    def parse_magnetic_shieldings(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> list['MagresParser.mag_shielding']:
        """
        Parse the magnetic shieldings from the magres file and assign `entity_ref` to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[MagresParser.mag_shielding]: The list of parsed `MagresParser.mag_shielding` sections.
        """

        # Ensure lookup is built
        if not hasattr(self, 'particle_lookup') or not self.particle_lookup:
            self.build_particle_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('ms', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning('The input magres file does not contain any `ms` data.')
            return []
        elif np.size(data) != n_atoms * (9 + 2):  # 2 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `ms` does not coincide with the number of atoms.'
            )
            return []

        # Parse magnetic shieldings and their refs to the specific `MagresParser.atom_state_class`
        magnetic_shieldings = []

        for atom_data in data:
            ps, values = self.extract_particle_state_and_tensor(atom_data, logger)
            sec_ms = self.mag_shielding(entity_ref=ps, indices=[ps._site_index])
            sec_ms.value = values * ureg('ppm')
            magnetic_shieldings.append(sec_ms)

        return magnetic_shieldings

    def parse_electric_field_gradients(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> 'MagresParser.e_field_gradient_class':
        """
        Parse the electric field gradients from the magres file and assign `entity_ref` to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            self.: The parsed `self.e_field_gradient_class` section.
        """
        # Ensure lookup is built
        if not hasattr(self, 'particle_lookup') or not self.particle_lookup:
            self.build_particle_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('efg', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning('The input magres file does not contain any `efg` data.')
            return []
        elif np.size(data) != n_atoms * (9 + 2):  # 2 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `efg` does not coincide with the number of atoms.'
            )
            return []

        # Parse electric field gradients and their refs to the specific `MagresParser.atom_state_class`
        electric_field_gradients = []

        for atom_data in data:
            ps, values = self.extract_particle_state_and_tensor(atom_data, logger)
            sec_efg = self.e_field_gradient_class(entity_ref=ps, indices=[ps._site_index])
            sec_efg.value = values * ureg('a_u_efg')
            electric_field_gradients.append(sec_efg)

        return electric_field_gradients

    def parse_indirect_spin_spin_couplings(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> list['MagresParser.indirect_spin_spin_couplings_class']:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[self.indirect_spin_spin_couplings_class]: The list of parsed `self.indirect_spin_spin_couplings_class` sections.
        """

        # Ensure lookup is built
        if not hasattr(self, 'particle_pair_lookup') or not self.particle_pair_lookup:
            self.build_particle_pair_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('isc', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning('The input magres file does not contain any `isc` data.')
            return []
        elif np.size(data) != n_atoms**2 * (9 + 4):  # 4 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `isc` does not coincide with the number of atoms.'
            )
            return []

        # Prepare output list of length n_atoms**2
        indirect_spin_spin_couplings = [None] * (n_atoms * n_atoms)

        for atom_data in data:
            pair, values = self.extract_particle_pair_and_tensor(atom_data, logger)
            ps1, ps2, i, j = pair
            sec_isc = self.indirect_spin_spin_couplings_class(
                entity_ref_1=ps1,
                entity_ref_2=ps2,
                indices=[ps1._site_index, ps2._site_index],
            )
            sec_isc.value = values * 1e19 * ureg('T^2/J')
            indirect_spin_spin_couplings[i * n_atoms + j] = sec_isc

        return indirect_spin_spin_couplings

    def parse_indirect_spin_spin_couplings_fc(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> list['MagresParser.indirect_spin_spin_couplings_fc_class']:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[self.indirect_spin_spin_couplings_fc_class]: The list of parsed `self.indirect_spin_spin_couplings_fc_class` sections.
        """
        # Ensure lookup is built
        if not hasattr(self, 'particle_pair_lookup') or not self.particle_pair_lookup:
            self.build_particle_pair_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('isc_fc', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning('The input magres file does not contain any `isc_fc` data.')
            return []
        elif np.size(data) != n_atoms**2 * (9 + 4):  # 4 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `isc_fc` does not coincide with the number of atoms.'
            )
            return []

        # Prepare output list of length n_atoms**2
        indirect_spin_spin_couplings_fermi_contact = [None] * (n_atoms * n_atoms)

        for atom_data in data:
            pair, values = self.extract_particle_pair_and_tensor(atom_data, logger)
            ps1, ps2, i, j = pair
            sec_isc_fc = self.indirect_spin_spin_couplings_fc_class(
                entity_ref_1=ps1,
                entity_ref_2=ps2,
                indices=[ps1._site_index, ps2._site_index],
            )
            sec_isc_fc.value = values * 1e19 * ureg('T^2/J')
            indirect_spin_spin_couplings_fermi_contact[i * n_atoms + j] = sec_isc_fc

        return indirect_spin_spin_couplings_fermi_contact

    def parse_indirect_spin_spin_couplings_orbital_d(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> list['MagresParser.indirect_spin_spin_couplings_orbital_d_class']:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[self.indirect_spin_spin_couplings_orbital_d_class]: The list of parsed `self.indirect_spin_spin_couplings_orbital_d_class` sections.
        """
        # Ensure lookup is built
        if not hasattr(self, 'particle_pair_lookup') or not self.particle_pair_lookup:
            self.build_particle_pair_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('isc_orbital_d', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning(
                'The input magres file does not contain any `isc_orbital_d` data.'
            )
            return []
        elif np.size(data) != n_atoms**2 * (9 + 4):  # 4 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `isc_orbital_d` does not coincide with the number of atoms.'
            )
            return []

        # Prepare output list of length n_atoms**2
        indirect_spin_spin_couplings_orbital_d = [None] * (n_atoms * n_atoms)

        for atom_data in data:
            pair, values = self.extract_particle_pair_and_tensor(atom_data, logger)
            ps1, ps2, i, j = pair
            sec_isc_orbital_d = self.indirect_spin_spin_couplings_orbital_d_class(
                entity_ref_1=ps1,
                entity_ref_2=ps2,
                indices=[ps1._site_index, ps2._site_index],
            )
            sec_isc_orbital_d.value = values * 1e19 * ureg('T^2/J')
            indirect_spin_spin_couplings_orbital_d[i * n_atoms + j] = sec_isc_orbital_d

        return indirect_spin_spin_couplings_orbital_d

    def parse_indirect_spin_spin_couplings_orbital_p(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> list['MagresParser.indirect_spin_spin_couplings_orbital_p_class']:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[self.indirect_spin_spin_couplings_orbital_p_class]: The list of parsed `self.indirect_spin_spin_couplings_orbital_p_class` sections.
        """
        # Ensure lookup is built
        if not hasattr(self, 'particle_pair_lookup') or not self.particle_pair_lookup:
            self.build_particle_pair_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('isc_orbital_p', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning(
                'The input magres file does not contain any `isc_orbital_p` data.'
            )
            return []
        elif np.size(data) != n_atoms**2 * (9 + 4):  # 4 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `isc_orbital_p` does not coincide with the number of atoms.'
            )
            return []

        # Prepare output list of length n_atoms**2
        indirect_spin_spin_couplings_orbital_p = [None] * (n_atoms * n_atoms)

        for atom_data in data:
            pair, values = self.extract_particle_pair_and_tensor(atom_data, logger)
            ps1, ps2, i, j = pair
            sec_isc_orbital_p = self.indirect_spin_spin_couplings_orbital_p_class(
                entity_ref_1=ps1,
                entity_ref_2=ps2,
                indices=[ps1._site_index, ps2._site_index],
            )
            sec_isc_orbital_p.value = values * 1e19 * ureg('T^2/J')
            indirect_spin_spin_couplings_orbital_p[i * n_atoms + j] = sec_isc_orbital_p

        return indirect_spin_spin_couplings_orbital_p

    def parse_indirect_spin_spin_couplings_spin(
        self,
        magres_data: TextParser,
        atom_state_class: 'MagresParser.atom_state_class',
        model_system: 'MagresParser.model_system_class',
        logger: 'BoundLogger',
    ) -> list['MagresParser.indirect_spin_spin_couplings_spin_class']:
        """
        Parse the spin-spin couplings from the magres file and assign `entity_ref_1` and `entity_ref_2`
        to the specific `MagresParser.atom_state_class`.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            cell ('MagresParser.cell_class'): The parsed `MagresParser.cell_class` section.
            atom_state_class ('MagresParser.atom_state_class'): The class representing the atom state section.
            model_system ('MagresParser.model_system_class'): The class representing the model system section.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[self.indirect_spin_spin_couplings_spin_class]: The list of parsed `self.indirect_spin_spin_couplings_spin_class` sections.
        """
        # Ensure lookup is built
        if not hasattr(self, 'particle_pair_lookup') or not self.particle_pair_lookup:
            self.build_particle_pair_lookup(model_system, logger)

        n_atoms = len(model_system.particle_states)
        data = magres_data.get('isc_spin', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning(
                'The input magres file does not contain any `isc_spin` data.'
            )
            return []
        elif np.size(data) != n_atoms**2 * (9 + 4):  # 4 extra columns with atom labels
            logger.warning(
                'The shape of the matched text from the magres file for the `isc_spin` does not coincide with the number of atoms.'
            )
            return []

        # Prepare output list of length n_atoms**2
        indirect_spin_spin_couplings_spin_dipolar = [None] * (n_atoms * n_atoms)

        for atom_data in data:
            pair, values = self.extract_particle_pair_and_tensor(atom_data, logger)
            ps1, ps2, i, j = pair
            sec_isc__spin_dipolar = self.indirect_spin_spin_couplings_spin_class(
                entity_ref_1=ps1,
                entity_ref_2=ps2,
                indices=[ps1._site_index, ps2._site_index],
            )
            sec_isc__spin_dipolar.value = values * 1e19 * ureg('T^2/J')
            indirect_spin_spin_couplings_spin_dipolar[i * n_atoms + j] = (
                sec_isc__spin_dipolar
            )

        return indirect_spin_spin_couplings_spin_dipolar

    def parse_magnetic_susceptibilities(
        self, magres_data: TextParser, logger: 'BoundLogger'
    ) -> list['MagresParser.mag_susceptibility_class']:
        """
        Parse the magnetic susceptibilities from the magres file.

        Args:
            magres_data (TextParser): The parsed [magres][/magres] block.
            logger (BoundLogger): The logger to log messages.

        Returns:
            list[MagresParser.mag_susceptibility_class]: The list of parsed `MagresParser.mag_susceptibility_class` sections.
        """

        data = magres_data.get('sus', [])

        # Initial check on the data block and size of the matched text
        if not data or np.size(data) == 0:
            logger.warning('The input magres file does not contain any `sus` data.')
            return []
        elif np.size(data) != 9:
            logger.warning(
                'The shape of the matched text from the magres file for the `sus` does not coincide with 9 (3x3 tensor).'
            )
            return []
        values = np.reshape(data, (3, 3))
        sec_sus = self.mag_susceptibility_class()
        sec_sus.value = values * ureg('m^3/mol')  # *1e-6 cm^3/mol
        return [sec_sus]

    def parse_outputs(
        self,
        simulation: 'MagresParser.simulation_class',
        atomsstate: 'MagresParser.atom_state_class',
        logger: 'BoundLogger',
    ) -> Optional['MagresParser.magres_outputs_class']:
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
                'Could not find the `MagresParser.model_system_class` that the outputs reference to.'
            )
            return None
        outputs = self.magres_outputs_class(
            model_method_ref=simulation.model_method[-1],
            model_system_ref=simulation.model_system[-1],
        )
        if not simulation.model_system[-1].particle_states:
            logger.warning(
                'Could not find the `MagresParser.atom_state_class` list under particle states.'
            )
            return None

        # Check if [magres][/magres] was correctly parsed
        magres_data = self.magres_file_parser.get('magres')
        if not magres_data:
            logger.warning('Could not find [magres] data block in magres file.')
            return None

        # Parse `MagresParser.mag_shielding`
        ms = self.parse_magnetic_shieldings(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(ms) > 0:
            outputs.magnetic_shieldings = ms

        # Parse `MagresParser.e_field_gradient_class`
        efg = self.parse_electric_field_gradients(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(efg) > 0:
            outputs.electric_field_gradients = efg

        # Parse `self.indirect_spin_spin_couplings_class`
        isc = self.parse_indirect_spin_spin_couplings(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(isc) > 0:
            outputs.indirect_spin_spin_couplings = isc

        # Parse `self.indirect_spin_spin_couplings_fc_class`
        isc_fc = self.parse_indirect_spin_spin_couplings_fc(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(isc_fc) > 0:
            outputs.indirect_spin_spin_couplings_fermi_contact = isc_fc

        # Parse `self.indirect_spin_spin_couplings_orbital_d_class`
        isc_orbital_d = self.parse_indirect_spin_spin_couplings_orbital_d(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(isc_orbital_d) > 0:
            outputs.indirect_spin_spin_couplings_orbital_d = isc_orbital_d

        # Parse `self.indirect_spin_spin_couplings_orbital_p_class`
        isc_orbital_p = self.parse_indirect_spin_spin_couplings_orbital_p(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(isc_orbital_p) > 0:
            outputs.indirect_spin_spin_couplings_orbital_p = isc_orbital_p

        # Parse `self.indirect_spin_spin_couplings_spin_class`
        isc_spin = self.parse_indirect_spin_spin_couplings_spin(
            magres_data=magres_data,
            atom_state_class=atomsstate,
            model_system=simulation.model_system[-1],
            logger=logger,
        )
        if len(isc_spin) > 0:
            outputs.indirect_spin_spin_couplings_spin_dipolar = isc_spin

        # Parse `MagresParser.mag_susceptibility_class`
        mag_sus = self.parse_magnetic_susceptibilities(
            magres_data=magres_data, logger=logger
        )
        if len(mag_sus) > 0:
            outputs.magnetic_susceptibilities = mag_sus

        return outputs

    def parse_nmr_magres_file_format(
        self, nmr_first_principles_archive: 'EntryArchive'
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
        workflow.name = 'NMR Magres'

        # ! Fix this once CASTEP and QuantumESPRESSO use the new `nomad-simulations` schema under 'data'
        # Method
        # method_nmr = extract_section(nmr_first_principles_archive, ['run', 'method'])
        # workflow.method.nmr_method_ref = method_nmr

        # Inputs and Outputs
        # ! Fix this to extract `input_structure` from `nmr_first_principles_archive` once
        # ! CASTEP and QuantumESPRESSO use the new `nomad-simulations` schema under 'data'
        input_structure = extract_section(self.archive, ['data', 'model_system'])
        nmr_magres_calculation = extract_section(self.archive, ['data', 'outputs'])
        if input_structure:
            workflow.m_add_sub_section(
                self.workflow_class.inputs,
                Link(name='Input structure', section=input_structure),
            )
        if nmr_magres_calculation:
            workflow.m_add_sub_section(
                self.workflow_class.outputs,
                Link(name='Output NMR calculation', section=nmr_magres_calculation),
            )

        # NMR (first principles) task
        # ! Fix this once CASTEP and QuantumESPRESSO use the new `nomad-simulations` schema under 'data'
        program_name = nmr_first_principles_archive.run[-1].program.name
        if nmr_first_principles_archive.workflow2:
            task = TaskReference(task=nmr_first_principles_archive.workflow2)
            task.name = f'NMR FirstPrinciples {program_name}'
            if input_structure:
                task.inputs = [Link(name='Input structure', section=input_structure)]
            if nmr_magres_calculation:
                task.outputs = [
                    Link(
                        name='Output NMR calculation',
                        section=nmr_magres_calculation,
                    )
                ]
            workflow.m_add_sub_section(self.workflow_class.tasks, task)

        self.archive.workflow2 = workflow

    def parse(
        self,
        filepath: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
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
        atom_state_class = self.atom_state_class
        calculation_params = self.magres_file_parser.get('calculation', {})
        code = calculation_params.get('code', '')

        # If code is a list as in QE-GIPAW cases, join it into a string and update the dict
        if isinstance(code, list):
            code = ' '.join(str(c) for c in code)
            calculation_params['code'] = code  # Update the dict with the fixed value
            logger.warning(
                'calc_code was parsed as a list, joining into string',
                result=code,
            )

        # Validate supported codes
        # TODO: Add support for other NMR codes as they become available
        supported_codes = ['CASTEP', 'QE']
        is_supported = any(supported in code for supported in supported_codes)

        if not is_supported:
            logger.error(
                'Only CASTEP and QE-GIPAW based NMR simulations are currently supported '
                'by the magres parser. Found calc_code: "%s"', code
            )
            return

        # Parse program information
        # Note: Older QE-GIPAW generated magres files may have limited metadata in the
        # calculation block, have incomplete or vague version information
        # (e.g., calc_code_version='git'). The parser attempts to extract version
        # from calc_code field when necessary.
        # TODO: Improve parsing when more recent QE-GIPAW files with richer metadata
        # become available.
        program_name, program_version = self._parse_program_info(
            calculation_params, logger
        )
        simulation.program = self.program_class(
            name=program_name,
            version=program_version,
        )

        # `MagresParser.model_system_class` parsing
        model_system = self.parse_model_system(logger=logger)
        if model_system is not None:
            simulation.model_system.append(model_system)

        # `MagresParser.model_method_class` parsing
        model_method = self.parse_model_method(calculation_params=calculation_params, logger=logger)
        simulation.model_method.append(model_method)

        # `self.magres_outputs` parsing
        outputs = self.parse_outputs(
            simulation=simulation, atomsstate=atom_state_class, logger=logger
        )
        if outputs is not None:
            simulation.outputs.append(outputs)

        archive.data = simulation
        # ! this will only work after the CASTEP and QE plugin parsers are defined
        # Try to resolve the `entry_id` and `mainfile` of other entries in the upload to connect the magres entry with the CASTEP or QuantumESPRESSO entry
        metadata = []
        try:
            from nomad.app.v1.models.models import MetadataRequired
            from nomad.search import search

            upload_id = self.archive.metadata.upload_id
            search_ids = search(
                owner='visible',
                user_id=self.archive.metadata.main_author.user_id,
                query={'upload_id': upload_id},
                required=MetadataRequired(include=['entry_id', 'mainfile']),
            ).data
            metadata = [[sid['entry_id'], sid['mainfile']] for sid in search_ids]
        except Exception:
            logger.warning(
                'Could not resolve the entry_id and mainfile of other entries in the upload.'
            )
            return
        for entry_id, mainfile in metadata:
            # if mainfile == filepath_stripped:  # we skip the current parsed mainfile
            #    continue
            # We try to load the archive from its context and connect both the CASTEP and the magres entries
            # ? add more checks on the system information for the connection?
            try:
                entry_archive = self.archive.m_context.load_archive(
                    entry_id, upload_id, None
                )
                # ! Fix this when CASTEP parser uses the new `data` schema
                method_label = entry_archive.run[-1].method[-1].label
                if method_label == 'NMR':
                    castep_archive = entry_archive
                    # We write the workflow MagresParser.workflow_class directly in the magres entry
                    self.parse_nmr_magres_file_format(
                        nmr_first_principles_archive=castep_archive
                    )
                    break
            except Exception:
                continue
