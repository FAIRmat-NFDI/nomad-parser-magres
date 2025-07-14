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
    labels = ['H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O']
    n_atoms = len(labels)
    for index, symbol in enumerate(labels):
        assert model_system.particle_states[index].chemical_symbol == symbol

    # Cell
    assert len(model_system.cell) == 1
    atomic_cell = model_system.cell[0]

    # Lattice vectors: atomic_cell.lattice_vectors is a numpy array (or Quantity) of shape (3, 3)
    expected_lattice = np.array(
        [
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0],
        ]
    )
    lattice_vectors = atomic_cell.lattice_vectors.to('angstrom').magnitude
    assert np.allclose(lattice_vectors, expected_lattice, atol=1e-8)

    assert atomic_cell.periodic_boundary_conditions == [True, True, True]

    # ModelMethod
    assert len(simulation.model_method) == 1
    dft = simulation.model_method[0]
    assert dft.m_def.name == 'DFT'
    assert dft.name == 'NMR'
    # XC functionals
    assert len(dft.xc_functionals) == 2
    # Order is correlation then exchange for PBE in the parser map
    assert dft.xc_functionals[0].name == 'correlation'
    assert dft.xc_functionals[0].libxc_name == 'GGA_C_PBE'
    assert dft.xc_functionals[1].name == 'exchange'
    assert dft.xc_functionals[1].libxc_name == 'GGA_X_PBE'
    # NumericalSettings
    assert len(dft.numerical_settings) == 1
    k_space = dft.numerical_settings[0]
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
        assert ms.entity_ref.chemical_symbol == labels[i]
        assert ms.entity_ref.label == labels[i]
        # Normalize to set the custom name
        ms.normalize(archive, logger)
        # The normalized name should be same as label[i]
        assert ms.name == f"{ms.entity_ref.label}"
    # Check tensor value
    assert np.isclose(
        output.magnetic_shieldings[3].value.magnitude,
        np.transpose(
            np.array(
                [
                    [3.22898154856e-5, 5.84330480731e-7, -1.63639006642e-6],
                    [7.78952021344e-7, 2.26711049351e-5, 1.80797334282e-6],
                    [-8.10936558433e-8, 2.01393309009e-6, 2.59791612443e-5],
                ]
            )
        ),
    ).all()
    # EFG Tensor - Check entity ref, site labels, and tensor value
    for i in range(n_atoms):
        efg = output.electric_field_gradients[i]
        assert efg.entity_ref.chemical_symbol == labels[i]
        assert efg.entity_ref.label == labels[i]
        # Normalize to set the custom name
        efg.normalize(archive, logger)
        # The normalized name should be same as label[i]
        assert efg.name == f"{efg.entity_ref.label}"
    # Check tensor value
    assert np.isclose(
        output.electric_field_gradients[3].value.magnitude,
        np.transpose(
            np.array(
                [
                    [1.25522205339, 0.60319384091, -1.83473060621],
                    [0.60319384091, -1.29847323513, -0.40585650492],
                    [-1.83473060621, -0.40585650492, 0.0432511817317],
                ]
            )
            * 9.717362e20
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
            assert isc.entity_ref_1.chemical_symbol == labels[i]
            assert isc.entity_ref_1.label == labels[i]
            assert isc.entity_ref_2.chemical_symbol == labels[j]
            assert isc.entity_ref_2.label == labels[j]
            # Normalize to set the custom name
            isc.normalize(archive, logger)
            # The normalized name should be 'label_i-label_j'
            assert isc.name == f"{labels[i]}-{labels[j]}"
    # Check tensor value
    assert np.isclose(
        output.indirect_spin_spin_couplings[3].value.magnitude,
        np.transpose(
            np.array(
                [
                    [0.10333898357, 0.00892814613052, 0.00789750719001],
                    [-0.00991323894355, 0.0689546520892, 0.0692766888876],
                    [-0.0147839319182, 0.060922336088, 0.0372243677199],
                ]
            )
            * 1e19
        ),
    ).all()
    # Check magnetic susceptibility is present and has correct length
    assert len(output.magnetic_susceptibilities) == 1  # 1 tensor for the system
    # Check tensor value
    assert np.isclose(
        output.magnetic_susceptibilities[0].value.magnitude,
        np.transpose(
            np.array(
                [
                    [-49.1163, -2.4718, -0.5854],
                    [-2.4603, -57.7785, -1.9644],
                    [-0.5680, -1.9538, -56.5384],
                ]
            )
        ),
    ).all()
