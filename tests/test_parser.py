import os

import numpy as np
import pytest
from nomad.datamodel import EntryArchive

from nomad_parser_magres.parsers.parser import MagresParser

from . import logger


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return MagresParser()


def test_single_point_ethanol(parser):
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'ethanol_nmr.magres'),
        archive,
        logger,
    )
    simulation = archive.data

    # Program
    assert simulation.program.name == 'CASTEP'
    assert simulation.program.version == '7.0'

    # ModelSystem
    assert len(simulation.model_system) == 1
    model_system = simulation.model_system[0]
    assert model_system.is_representative
    # Positions: model_system.positions is a numpy array (or Quantity) of shape (n_atoms, 3)
    positions = model_system.positions.to('angstrom').magnitude
    expected_position = np.array([-1.130705, 0.974874, 1.564773])
    assert np.allclose(positions[3], expected_position, atol=1e-6)
    # Particle States
    assert len(model_system.particle_states) == 9
    labels = ['H1_1', 'H1_2', 'H1_3', 'H2_4', 'H2_5', 'H2_6', 'C1_1', 'C2_2', 'O1_1']
    chemical_symbols = ['H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O']
    n_atoms = len(labels)
    for index, symbol in enumerate(chemical_symbols):
        assert model_system.particle_states[index].chemical_symbol == symbol
    for index, label in enumerate(labels):
        assert model_system.particle_states[index].label == label

    # Lattice vectors: model_system.lattice_vectors is a numpy array (or Quantity) of shape (3, 3)
    expected_lattice = np.array(
        [
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0],
        ]
    )
    lattice_vectors = model_system.lattice_vectors.to('angstrom').magnitude
    assert np.allclose(lattice_vectors, expected_lattice, atol=1e-8)

    assert model_system.periodic_boundary_conditions == [True, True, True]

    # ModelMethod
    assert len(simulation.model_method) == 1
    dft = simulation.model_method[0]
    assert dft.m_def.name == 'DFT'
    assert dft.name == 'NMR'
    # XC functional - single XCFunctional object (not a list)
    assert dft.xc is not None
    assert dft.xc.functional_key == 'PBE'
    # NumericalSettings: BasisSetContainer (index 0) + KSpace (index 1)
    assert len(dft.numerical_settings) == 2
    # Basis set (cutoff energy)
    basis_container = dft.numerical_settings[0]
    assert basis_container.m_def.name == 'BasisSetContainer'
    assert len(basis_container.basis_set_components) == 1
    pw_basis = basis_container.basis_set_components[0]
    assert pw_basis.m_def.name == 'PlaneWaveBasisSet'
    # ethanol: cutoffenergy = 40.0 Hartree -> 1088.46 eV
    assert np.isclose(pw_basis.cutoff_energy.to('eV').magnitude, 1088.455449839241, rtol=1e-6)
    # KSpace
    k_space = dft.numerical_settings[1]
    assert k_space.m_def.name == 'KSpace'
    # KMesh
    assert len(k_space.k_mesh) == 1
    assert (np.array(k_space.k_mesh[0].grid) == [1, 1, 1]).all()
    assert np.allclose(k_space.k_mesh[0].offset, [0.25, 0.25, 0.25], atol=1e-8)

    # Outputs
    assert len(simulation.outputs) == 1
    output = simulation.outputs[0]
    assert output.model_system_ref == model_system
    assert output.model_method_ref == dft
    # Magnetic Shielding and Electric Field Gradient
    # Check tensors are non-empty and have correct length
    assert len(output.magnetic_shieldings) == 9  # 6 H, 2 C, 1 O
    assert len(output.electric_field_gradients) == 9
    # Magnetic Shielding - Check entity ref, site labels, and tensor value
    for i in range(n_atoms):
        ms = output.magnetic_shieldings[i]
        assert ms.entity_ref.chemical_symbol == chemical_symbols[i]
        assert ms.entity_ref.label == labels[i]
        # Normalize to set the custom name
        ms.normalize(archive, logger)
        # The normalized name should be same as label[i]
        assert ms.name == ms.entity_ref.label
    # Check tensor value (now stored in ppm units)
    assert np.isclose(
        output.magnetic_shieldings[3].value.magnitude,
        np.array(
                [
                    [32.2898154856, 0.584330480731, -1.63639006642],
                    [0.778952021344, 22.6711049351, 1.80797334282],
                    [-0.0810936558433, 2.01393309009, 25.9791612443],
                ]
            ),
    ).all()
    # EFG Tensor - Check entity ref, site labels, and tensor value
    for i in range(n_atoms):
        efg = output.electric_field_gradients[i]
        assert efg.entity_ref.chemical_symbol == chemical_symbols[i]
        assert efg.entity_ref.label == labels[i]
        # Normalize to set the custom name
        efg.normalize(archive, logger)
        # The normalized name should be same as label[i]
        assert efg.name == f"{efg.entity_ref.label}"
    # Check tensor value (now stored in Hartree atomic units)
    assert np.isclose(
        output.electric_field_gradients[3].value.magnitude,
        np.array(
                [
                    [0.125522205339, 0.060319384091, -0.183473060621],
                    [0.060319384091, -0.129847323513, -0.040585650492],
                    [-0.183473060621, -0.040585650492, 0.00432511817317],
                ]
            ),
    ).all()
    # Check spin-spin coupling contributions are present and have correct length
    assert (
        len(output.indirect_spin_spin_couplings) == 9**2
    )  # 9 atoms, each with 9 contributions
    assert len(output.indirect_spin_spin_couplings_fermi_contact) == 9**2
    assert len(output.indirect_spin_spin_couplings_orbital_d) == 9**2
    assert len(output.indirect_spin_spin_couplings_orbital_p) == 9**2
    assert len(output.indirect_spin_spin_couplings_spin_dipolar) == 9**2

    # Check entity_refs, site labels and tensor value for
    # indirect spin-spin couplings
    for i in range(n_atoms):
        for j in range(n_atoms):
            idx = i * n_atoms + j
            isc = output.indirect_spin_spin_couplings[idx]
            # Check the chemical symbols and labels for both entities
            assert isc.entity_ref_1.chemical_symbol == chemical_symbols[i]
            assert isc.entity_ref_1.label == labels[i]
            assert isc.entity_ref_2.chemical_symbol == chemical_symbols[j]
            assert isc.entity_ref_2.label == labels[j]
            # Normalize to set the custom name
            isc.normalize(archive, logger)
            # The normalized name should be 'label_i-label_j'
            assert isc.name == f"{labels[i]}-{labels[j]}"
    # Check tensor value
    assert np.isclose(
        output.indirect_spin_spin_couplings[3].value.magnitude,
        np.array(
                [
                    [0.10333898357, 0.00892814613052, 0.00789750719001],
                    [-0.00991323894355, 0.0689546520892, 0.0692766888876],
                    [-0.0147839319182, 0.060922336088, 0.0372243677199],
                ]
            )
            * 1e19,  # Convert to T^2/J
    ).all()
    # Check magnetic susceptibility is present and has correct length
    assert len(output.magnetic_susceptibilities) == 1  # 1 tensor for the system
    # Check tensor value
    assert np.isclose(
        output.magnetic_susceptibilities[0].value.magnitude,
        np.array(
                [
                    [-49.1163, -2.4718, -0.5854],
                    [-2.4603, -57.7785, -1.9644],
                    [-0.5680, -1.9538, -56.5384],
                ]
            ),
    ).all()


# ---------------------------------------------------------------------------
# Cutoff-energy parsing – full coverage checklist
# ---------------------------------------------------------------------------
# Each test maps to one cell of the 2×4 matrix:
#   code        × (unit storage)  × (notation)
#
#  ID  Description                                    File
#  [1] CASTEP  inline eV      decimal    (real)       silicon.magres
#  [2] CASTEP  inline eV      scientific (synthetic)  castep_cutoff_ev_scientific.magres
#  [3] CASTEP  separate Ha    decimal    (synthetic)  castep_cutoff_hartree_decimal.magres
#  [4] CASTEP  separate Ha    scientific (real)       ethanol_nmr.magres  ← test_single_point_ethanol
#  [5] QE      inline Ry      decimal    (real)       quartz.nmr.magres
#  [6] QE      inline Ry      scientific (synthetic)  qe_cutoff_ry_scientific.magres
#  [7] no-unit decimal        (error path, synthetic) no_unit_cutoff.magres
#  [8] no-unit scientific     (error path, synthetic) no_unit_cutoff_scientific.magres
#
# Unit conversion references (pint):
#   1 Hartree = 27.211386245988 eV
#   1 Rydberg = 13.60569225 eV
# ---------------------------------------------------------------------------

def _get_cutoff_ev(dft):
    """Return cutoff energy in eV from the first BasisSetContainer found, or None."""
    for ns in dft.numerical_settings:
        if ns.m_def.name == 'BasisSetContainer':
            return ns.basis_set_components[0].cutoff_energy.to('eV').magnitude
    return None


# [1] CASTEP inline eV decimal (silicon.magres)
def test_cutoff_inline_ev(parser):
    """[1] CASTEP inline eV decimal: 250.00000000 eV  →  250.0 eV."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'silicon.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    cutoff_ev = _get_cutoff_ev(dft)
    assert cutoff_ev is not None, 'BasisSetContainer not found – cutoff was not parsed'
    assert np.isclose(cutoff_ev, 250.0, rtol=1e-4)


# [2] CASTEP inline eV scientific (synthetic file created)
def test_cutoff_castep_ev_scientific(parser):
    """[2] CASTEP inline eV scientific: 2.50000000E+02 eV  →  250.0 eV."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'castep_cutoff_ev_scientific.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    cutoff_ev = _get_cutoff_ev(dft)
    assert cutoff_ev is not None, 'BasisSetContainer not found – cutoff was not parsed'
    assert np.isclose(cutoff_ev, 250.0, rtol=1e-4)


# [3] CASTEP separate Hartree decimal (synthetic file created)
def test_cutoff_castep_hartree_decimal(parser):
    """[3] CASTEP separate Hartree declaration, decimal: 40.0 Ha  →  ~1088.46 eV."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'castep_cutoff_hartree_decimal.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    cutoff_ev = _get_cutoff_ev(dft)
    assert cutoff_ev is not None, 'BasisSetContainer not found – cutoff was not parsed'
    # 1 Ha = 27.211386245988 eV
    assert np.isclose(cutoff_ev, 40.0 * 27.211386245988, rtol=1e-4)


# [4] CASTEP separate Hartree scientific → covered by test_single_point_ethanol


# [5] QE inline Ry decimal (quartz.nmr.magres)
def test_cutoff_inline_ry(parser):
    """[5] QE inline Ry decimal: 40.00 Ry  →  40 × 13.6057 ≈ 544.23 eV."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'quartz.nmr.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    cutoff_ev = _get_cutoff_ev(dft)
    assert cutoff_ev is not None, 'BasisSetContainer not found – cutoff was not parsed'
    # 1 Ry = 13.60569225 eV
    assert np.isclose(cutoff_ev, 40.0 * 13.60569225, rtol=1e-4)


# [6] QE inline Ry scientific (synthetic file created)
def test_cutoff_qe_ry_scientific(parser):
    """[6] QE inline Ry scientific: 4.00000E+01 Ry  →  40 × 13.6057 ≈ 544.23 eV."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'qe_cutoff_ry_scientific.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    cutoff_ev = _get_cutoff_ev(dft)
    assert cutoff_ev is not None, 'BasisSetContainer not found – cutoff was not parsed'
    assert np.isclose(cutoff_ev, 40.0 * 13.60569225, rtol=1e-4)


# [7] No unit – decimal value (error path)
def test_cutoff_no_unit(parser):
    """[7] Decimal value only, no unit declared: must warn and skip BasisSetContainer."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'no_unit_cutoff.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    assert _get_cutoff_ev(dft) is None, (
        'BasisSetContainer should NOT be populated when no unit is declared'
    )


# [8] No unit – scientific value (error path)
def test_cutoff_no_unit_scientific(parser):
    """[8] Scientific value only, no unit declared: must warn and skip BasisSetContainer."""
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'no_unit_cutoff_scientific.magres'),
        archive,
        logger,
    )
    dft = archive.data.model_method[0]
    assert _get_cutoff_ev(dft) is None, (
        'BasisSetContainer should NOT be populated when no unit is declared'
    )
