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
    assert simulation.program.version == '24.1'

    # ModelSystem
    assert len(simulation.model_system) == 1
    model_system = simulation.model_system[0]
    assert model_system.is_representative
    #   Cell
    assert len(model_system.cell) == 1
    atomic_cell = model_system.cell[0]
    assert np.isclose(
        atomic_cell.positions[3].to('angstrom').magnitude,
        np.array([3.57828732, 5.39462129, 5.22149125]),
    ).all()
    assert np.isclose(
        atomic_cell.lattice_vectors.to('angstrom').magnitude,
        np.array(
            [
                [5.29177211e00, 0.00000000e00, 0.00000000e00],
                [3.24027589e-16, 5.29177211e00, 0.00000000e00],
                [3.24027589e-16, 3.24027589e-16, 5.29177211e00],
            ]
        ),
    ).all()
    assert atomic_cell.periodic_boundary_conditions == [True, True, True]
    #       AtomsState
    assert len(atomic_cell.atoms_state) == 9
    labels = ['H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O']
    for index, symbol in enumerate(labels):
        assert atomic_cell.atoms_state[index].chemical_symbol == symbol

    # ModelMethod
    assert len(simulation.model_method) == 1
    assert simulation.model_method[0].m_def.name == 'DFT'
    assert simulation.model_method[0].name == 'NMR'
    dft = simulation.model_method[0]
    assert len(dft.xc_functionals) == 2
    assert dft.xc_functionals[0].name == 'correlation'
    assert dft.xc_functionals[0].libxc_name == 'LDA_C_PZ'
    assert dft.xc_functionals[1].name == 'exchange'
    assert dft.xc_functionals[1].libxc_name == 'LDA_X_PZ'
    #   NumericalSettings
    assert len(dft.numerical_settings) == 1
    assert dft.numerical_settings[0].m_def.name == 'KSpace'
    k_space = dft.numerical_settings[0]
    #       KMesh
    assert len(k_space.k_mesh) == 1
    assert (k_space.k_mesh[0].grid == [1, 1, 1]).all()
    assert (k_space.k_mesh[0].offset == [0.25, 0.25, 0.25]).all()

    # Outputs
    assert len(simulation.outputs) == 1
    output = simulation.outputs[0]
    assert output.model_system_ref == model_system
    assert output.model_method_ref == dft
    #   Properties
    assert len(output.m_xpath('magnetic_shieldings', dict=False)) == 9  # per atom
    for property_name in [
        'electric_field_gradients',
        'magnetic_shieldings',
    ]:
        assert output.m_xpath(property_name, dict=False) is not None
    for property_name in [
        'spin_spin_couplings',
        'magnetic_susceptibilities',
    ]:
        assert output.m_xpath(property_name, dict=False) is None
    #       MagneticShieldingTensor
    for i, ms in enumerate(output.magnetic_shieldings):
        assert ms.entity_ref.chemical_symbol == labels[i]
    assert np.isclose(
        output.magnetic_shieldings[3].value.magnitude,
        np.array(
            [
                [3.15771355e-05, -5.88661144e-07, 1.53864065e-06],
                [-4.68026860e-07, 2.06392827e-05, 2.43151206e-06],
                [7.98507383e-08, 9.14578022e-07, 2.48414650e-05],
            ]
        ),
    ).all()
    assert np.isclose(
        output.electric_field_gradients[0].efg_total[5].value.magnitude,
        np.array(
            [
                [
                    -1.3166582037591257e21,
                    1.30299532627198460000e20,
                    1.9811002968672890000e19,
                ],
                [
                    1.30299532627198460000e20,
                    1.6767151575540092e21,
                    -1.644761535344864e21,
                ],
                [
                    1.9811002968672890000e19,
                    -1.644761535344864e21,
                    -3.60056953794901300000e20,
                ],
            ]
        ),
    ).all()



def test_quartz_nmr(parser):
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'quartz.nmr.magres'),
        archive,
        logger,
    )
    simulation = archive.data

    from devtools import debug

    # Program
    assert simulation.program.name == 'QE'
    assert simulation.program.version == '7.4.1'

    # ModelSystem
    assert len(simulation.model_system) == 1
    model_system = simulation.model_system[0]
    assert model_system.is_representative
    #   Cell
    assert len(model_system.cell) == 1
    atomic_cell = model_system.cell[0]

    assert np.isclose(
        atomic_cell.positions[3].to('angstrom').magnitude,
        np.array([1.673406, -0.623251,  1.158539]),
    ).all()
    assert np.isclose(
        atomic_cell.lattice_vectors.to('angstrom').magnitude,
        np.array(
            [
                [ 2.456196, -4.254274,  0.      ],
                [ 2.456196,  4.254274,  0.      ],
                [ 0.,        0.,        5.403631],
            ]
        ),
    ).all()
    assert atomic_cell.periodic_boundary_conditions == [True, True, True]
    #       AtomsState
    assert len(atomic_cell.atoms_state) == 9
    labels = ['Si', 'Si', 'Si', 'O', 'O', 'O', 'O', 'O', 'O']
    for index, symbol in enumerate(labels):
        assert atomic_cell.atoms_state[index].chemical_symbol == symbol

    # ModelMethod
    assert len(simulation.model_method) == 1
    assert simulation.model_method[0].m_def.name == 'DFT'
    assert simulation.model_method[0].name == 'NMR'
    dft = simulation.model_method[0]
    assert len(dft.xc_functionals) == 2
    assert dft.xc_functionals[0].name == 'correlation'
    assert dft.xc_functionals[0].libxc_name == 'GGA_C_PBE'
    assert dft.xc_functionals[1].name == 'exchange'
    assert dft.xc_functionals[1].libxc_name == 'GGA_X_PBE'
    #   NumericalSettings
    assert len(dft.numerical_settings) == 1
    assert dft.numerical_settings[0].m_def.name == 'KSpace'
    k_space = dft.numerical_settings[0]
    #       KMesh
    assert len(k_space.k_mesh) == 1
    assert (k_space.k_mesh[0].grid == [4, 4, 4]).all()
    assert (k_space.k_mesh[0].offset == [0.5, 0.5, 0.5]).all()

    # Outputs
    assert len(simulation.outputs) == 1
    output = simulation.outputs[0]
    assert output.model_system_ref == model_system
    assert output.model_method_ref == dft
    #   Properties
    assert len(output.m_xpath('magnetic_shieldings', dict=False)) == 9  # per atom
    for property_name in [
        'magnetic_shieldings',
        'magnetic_susceptibilities',
    ]:
        assert output.m_xpath(property_name, dict=False) is not None
    for property_name in [
        'electric_field_gradients',
        'spin_spin_couplings',
    ]:
        assert output.m_xpath(property_name, dict=False) is None


    
def test_quartz_efg(parser):
    archive = EntryArchive()
    parser.parse(
        os.path.join('tests', 'data', 'quartz.efg.magres'),
        archive,
        logger,
    )
    simulation = archive.data

    from devtools import debug

    # Program
    assert simulation.program.name == 'QE'
    assert simulation.program.version == '7.4.1'

    # ModelSystem
    assert len(simulation.model_system) == 1
    model_system = simulation.model_system[0]
    assert model_system.is_representative
    #   Cell
    assert len(model_system.cell) == 1
    atomic_cell = model_system.cell[0]

    assert np.isclose(
        atomic_cell.positions[3].to('angstrom').magnitude,
        np.array([1.673406, -0.623251,  1.158539]),
    ).all()
    assert np.isclose(
        atomic_cell.lattice_vectors.to('angstrom').magnitude,
        np.array(
            [
                [ 2.456196, -4.254274,  0.      ],
                [ 2.456196,  4.254274,  0.      ],
                [ 0.,        0.,        5.403631],
            ]
        ),
    ).all()
    assert atomic_cell.periodic_boundary_conditions == [True, True, True]
    #       AtomsState
    assert len(atomic_cell.atoms_state) == 9
    labels = ['Si', 'Si', 'Si', 'O', 'O', 'O', 'O', 'O', 'O']
    for index, symbol in enumerate(labels):
        assert atomic_cell.atoms_state[index].chemical_symbol == symbol

    # ModelMethod
    assert len(simulation.model_method) == 1
    assert simulation.model_method[0].m_def.name == 'DFT'
    assert simulation.model_method[0].name == 'NMR'
    dft = simulation.model_method[0]
    assert len(dft.xc_functionals) == 2
    assert dft.xc_functionals[0].name == 'correlation'
    assert dft.xc_functionals[0].libxc_name == 'GGA_C_PBE'
    assert dft.xc_functionals[1].name == 'exchange'
    assert dft.xc_functionals[1].libxc_name == 'GGA_X_PBE'
    #   NumericalSettings
    assert len(dft.numerical_settings) == 1
    assert dft.numerical_settings[0].m_def.name == 'KSpace'
    k_space = dft.numerical_settings[0]
    #       KMesh
    assert len(k_space.k_mesh) == 1
    assert (k_space.k_mesh[0].grid == [4, 4, 4]).all()
    assert (k_space.k_mesh[0].offset == [0.5, 0.5, 0.5]).all()

    # Outputs
    assert len(simulation.outputs) == 1
    output = simulation.outputs[0]
    assert output.model_system_ref == model_system
    assert output.model_method_ref == dft
    #   Properties
    assert output.electric_field_gradients is not None
    for property_name in [
        'magnetic_susceptibilities',
        'magnetic_shieldings',
        'spin_spin_couplings',
    ]:
        assert output.m_xpath(property_name, dict=False) is None