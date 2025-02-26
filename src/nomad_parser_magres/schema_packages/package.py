from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from nomad.metainfo import Context, Section
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.datamodel.metainfo.basesections import Entity
from nomad.metainfo import MEnum, Quantity, SchemaPackage, Section, SubSection
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.outputs import Outputs as BaseOutputs
from nomad_simulations.schema_packages.physical_property import PhysicalProperty


m_package = SchemaPackage()


def resolve_name_from_entity_ref(entities: list[Entity], logger: "BoundLogger") -> str:
    """
    Resolves the `name` of the atom-resolved `PhysicalProperty` from the `entity_ref` by assigning
    a label corresponding to the `AtomsState.chemical_symbol` and a number corresponding to the
    position in the list of `AtomsState`.

    Args:
        entities (list[Entity]): The list of entities to resolve the name from.
        logger ('BoundLogger'): The logger to log messages.

    Returns:
        (str): The resolved name of the atom-resolved `PhysicalProperty`.
    """
    name = ""
    for entity in entities:
        atoms_state = entity
        # Check if `entity_ref` exists and it is an AtomsState
        if not atoms_state or not isinstance(atoms_state, AtomsState):
            logger.error(
                "Could not find `entity_ref` referencing an `AtomsState` section."
            )
            return ""
        # Check if the parent of `entity_ref` exists
        cell = atoms_state.m_parent
        if not cell:
            logger.warning(
                "The parent of the `AtomsState` in `entity_ref` does not exist."
            )
            return ""

        index = ""  # ! implement here if needed
        name += f"{atoms_state.chemical_symbol}{index}"
    return name


class MagneticShieldingTensor(PhysicalProperty):
    """
    Nuclear response of a material to shield the effects of an applied external field. This is a tensor 3x3 related with
    the induced magnetic field as:

        B_induced = - magnetic_shielding * B_external

    See, e.g, https://pubs.acs.org/doi/10.1021/cr300108a.

    This property will appear as a list under `Outputs` where each of the elements correspond to an atom in the unit cell
    The specific atom is known by defining the reference to the specific `AtomsState` under `ModelSystem.cell.atoms_state`
    using `entity_ref`.
    """

    value = Quantity(
        type=np.float64,
        shape=[3, 3],
        unit="dimensionless",
        description="""
        Value of the magnetic shielding tensor per atom.
        """,
    )
    value_isotropic = Quantity(
        type=np.float64,
        unit="dimensionless",
        description="""
            The isotropic part of the `MagneticShieldingTensor`. This is 1/3 of the trace of the magnetic
            shielding tensor (see `extract_isotropic_part()` function in `MagneticShieldingTensor`).

            See, e.g, https://pubs.acs.org/doi/10.1021/cr300108a.
        """,
    )

    def __init__(
        self, m_def: "Section" = None, m_context: "Context" = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! this info is in the shape attribute of the Quantity
        self.name = self.m_def.name

    def extract_isotropic_part(
        self, logger: "BoundLogger"
    ) -> Optional[float]:
        """
        Extract the isotropic part of the magnetic shielding tensor. This is 1/3 of the trace of the magnetic
        shielding tensor `value`.

        Args:
            logger ('BoundLogger'): The logger to log messages.

        Returns:
            (Optional[float]): The isotropic part of the magnetic shielding tensor.
        """
        try:
            # Calculate the isotropic value
            isotropic = np.trace(np.array(self.value)) / 3.0
        except Exception:
            logger.warning("Could not extract the trace of the `value` tensor.")
            return None
        return isotropic

    def normalize(self, archive: "EntryArchive", logger: "BoundLogger") -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )

        # isotropic value extraction
        isotropic = self.extract_isotropic_part(logger)
        if isotropic is not None:
            logger.info(f"Appending isotropic value for {self.name}")
            self.value_isotropic = isotropic
        else:
            logger.warning(f"Isotropic value extraction failed for {self.name}")


class ElectricFieldGradient(PhysicalProperty):
    """
    Interaction between the quadrupole moment of the nucleus and the electric field gradient (EFG)
    at the nucleus position generated by the surrounding charges. This property is relevant for
    Nuclear Magnetic Resonance (NMR). The eigenvalues of these tensors can be used to compute
    the `quadrupolar_coupling_constant` and the `asymmetry_parameter`.

    See, e.g, https://pubs.acs.org/doi/10.1021/cr300108a.

    This property will appear as a list under `Outputs` where each of the elements correspond to an atom in the unit cell
    The specific atom is known by defining the reference to the specific `AtomsState` under `ModelSystem.cell.atoms_state`
    using `entity_ref`.
    """

    type = Quantity(
        type=MEnum("total", "local", "non_local"),
        description="""
        Type of contribution to the electric field gradient (EFG). The total EFG can be decomposed
        on the `local` and `non_local` contributions.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit="volt / meter ** 2",
        description="""
        Value of the electric field gradient (EFG) for each `contribution` per unit area.
        """,
    )

    quadrupolar_coupling_constant = Quantity(
        type=np.float64,
        description="""
        Quadrupolar coupling constant for each atom in the unit cell. It is computed from
        the eigenvalues of the EFG tensor as:

            quadrupolar_coupling_constant = efg_zz * e * Z / h

        where efg_zz is the largest eigenvalue of the EFG tensor, Z is the atomic number.
        """,
    )

    asymmetry_parameter = Quantity(
        type=np.float64,
        description="""
        Asymmetry parameter for each atom in the unit cell. It is computed from the
        eigenvalues of the EFG tensor as:

            asymmetry_parameter = (efg_xx - efg_yy) / efg_zz

        where efg_xx, efg_yy and efg_zz are the eigenvalues of the EFG tensor ordered
        such that |efg_zz| > |efg_yy| > |efg_xx|.
        """,
    )

    def __init__(
        self, m_def: "Section" = None, m_context: "Context" = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions
        self.name = self.m_def.name

    def resolve_quadrupolar_coupling_constant(self, logger: "BoundLogger") -> None:
        pass

    def resolve_asymmetry_parameter(self, logger: "BoundLogger") -> None:
        pass

    def normalize(self, archive: "EntryArchive", logger: "BoundLogger") -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref], logger=logger
        )

        # TODO add normalization to extract `quadrupolar_coupling_constant` and `asymmetry_parameter`


class ElectricFieldGradients(BaseOutputs):
    """
    Represents the electric field gradients (EFG) data extracted from a magres file.

    This class contains the total, local, and non-local components of the electric field gradients.
    Each component is represented as a subsection of the ElectricFieldGradient class and can have multiple entries.

    Attributes:
        efg_total (SubSection): A list of total electric field gradient entries.
        efg_local (SubSection): A list of local electric field gradient entries.
        efg_nonlocal (SubSection): A list of non-local electric field gradient entries.
    """

    efg_total = SubSection(sub_section=ElectricFieldGradient.m_def, repeats=True)
    efg_local = SubSection(sub_section=ElectricFieldGradient.m_def, repeats=True)
    efg_nonlocal = SubSection(sub_section=ElectricFieldGradient.m_def, repeats=True)


class SpinSpinCoupling(PhysicalProperty):
    """
    Indirect exchanges or interactions between 2 nuclear spins that arises from hyperfine interactions between
    the nuclei and local electrons.

    This property will appear as a list under `Outputs` where each of the elements correspond to an atom-atom
    coupling term. The specific pair of atoms defined for the coupling is known by referencing the specific `AtomsState`
    under `ModelSystem.cell.atoms_state` using `entity_ref_1` and `entity_ref_2`.

    Synonyms:
        - IndirectSpinSpinCoupling
    """

    # TODO extend this to other spin-spin coupling types besides indirect (which is useful in NMR)

    # we hide `entity_ref` from `PhysicalProperty` to avoid confusion
    m_def = Section(a_eln={"hide": ["entity_ref"]})

    type = Quantity(
        type=MEnum(
            "total",
            "direct_dipolar",
            "fermi_contact",
            "orbital_diamagnetic",
            "orbital_paramagnetic",
            "spin_dipolar",
        ),
        description="""
        Type of contribution to the indirect spin-spin coupling. The total indirect spin-spin
        coupling is composed of:

            `total` = `direct_dipolar` + J_coupling

        Where the J_coupling is:
            J_coupling = `fermi_contact`
                        + `spin_dipolar`
                        + `orbital_diamagnetic`
                        + `orbital_paramagnetic`

        See https://pubs.acs.org/doi/full/10.1021/cr300108a.
        """,
    )

    value = Quantity(
        type=np.float64,
        unit="joule",
        description="""
        Value of the indirect spin-spin couplings for each contribution.
        """,
    )

    reduced_value = Quantity(
        type=np.float64,
        unit="kelvin**2 / joule",
        shape=[3, 3],  # dynamical shape only works for `PhysicalProperty.value`
        description="""
        Reduced value of the indirect spin-spin couplings for each contribution. It relates with the
        normal value as:

            reduced_value = value / (gyromagnetic_ratio_i * gyromagnetic_ratio_j * 2 * np.pi * hbar)

        where i, j runs for each atom in the unit cell.
        """,
    )

    entity_ref_1 = Quantity(
        type=Entity,
        description="""
        Reference to the first entity that the coupling refers to. In this case, this is the
        first `AtomsState` in the pair of atoms that the coupling refers to.
        """,
    )

    entity_ref_2 = Quantity(
        type=Entity,
        description="""
        Reference to the second entity that the coupling refers to. In this case, this is the
        second `AtomsState` in the pair of atoms that the coupling refers to.
        """,
    )

    def __init__(
        self, m_def: "Section" = None, m_context: "Context" = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions

    def resolve_reduced_value(self, logger: "BoundLogger") -> None:
        pass

    def normalize(self, archive: "EntryArchive", logger: "BoundLogger") -> None:
        super().normalize(archive, logger)

        # Resolve `name` to be from the `entity_ref`
        self.name = resolve_name_from_entity_ref(
            entities=[self.entity_ref_1, self.entity_ref_2], logger=logger
        )

        # TODO add normalization to extract `value` from `reduced_value`
        # TODO add normalization to extract `reduced_value` from `value`


class MagneticSusceptibility(PhysicalProperty):
    """
    Section containing the information of magnetic susceptibility tensor. Degree of
    magnetization of a material in the presence of a magnetic field.
    """

    # TODO currently only the macroscopic quantity is being supported

    m_def = Section(validate=False)

    scale_dimension = Quantity(
        type=MEnum("microscopic", "macroscopic"),
        description="""
        Identifier of the scale dimension of the magnetic susceptibility tensor.
        """,
    )

    value = Quantity(  # TODO extend this to microscopic contributions
        type=np.float64,
        unit="dimensionless",
        description="""
        Value of the magnetic susceptibility tensor.
        """,
    )

    def __init__(
        self, m_def: "Section" = None, m_context: "Context" = None, **kwargs
    ) -> None:
        super().__init__(m_def, m_context, **kwargs)
        self.rank = [3, 3]  # ! move this to definitions

    def normalize(self, archive: "EntryArchive", logger: "BoundLogger") -> None:
        super().normalize(archive, logger)


class Outputs(BaseOutputs):
    """
    The outputs of the magres file format.
    """

    magnetic_shieldings = SubSection(
        sub_section=MagneticShieldingTensor.m_def, repeats=True
    )
    electric_field_gradients = SubSection(
        sub_section=ElectricFieldGradient.m_def, repeats=True
    )
    spin_spin_couplings = SubSection(sub_section=SpinSpinCoupling.m_def, repeats=True)
    magnetic_susceptibilities = SubSection(
        sub_section=MagneticSusceptibility.m_def, repeats=True
    )


m_package.__init_metainfo__()
