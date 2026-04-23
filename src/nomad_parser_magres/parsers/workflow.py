from nomad.metainfo import Quantity, Reference, SubSection
from nomad_simulations.schema_packages.model_method import ModelMethod
from simulationworkflowschema import (
    SerialSimulation,
    SimulationWorkflowMethod,
    SimulationWorkflowResults,
)


class NMRMagResResults(SimulationWorkflowResults):
    """
    Groups the NMR magres outputs.
    """

    pass


class NMRMagResMethod(SimulationWorkflowMethod):
    """
    References the NMR (first principles) input methodology.
    """

    nmr_method_ref = Quantity(
        type=Reference(ModelMethod),
        description="""
        Reference to the NMR (first principles) methodology.
        """,
    )


class NMRMagRes(SerialSimulation):
    """
    The NMR MagRes workflow is generated in an extra EntryArchive IF both the NMR (first
    principles) and the NMR magres SinglePoint EntryArchives are present in the
    upload.
    """

    method = SubSection(sub_section=NMRMagResMethod)

    results = SubSection(sub_section=NMRMagResResults)

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
