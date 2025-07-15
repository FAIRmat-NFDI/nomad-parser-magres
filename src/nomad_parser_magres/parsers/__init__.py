from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field


class MagresParserEntryPoint(ParserEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_parser_magres.parsers.parser import MagresParser

        return MagresParser(**self.dict())


magres_parser = MagresParserEntryPoint(
    name='MagresParserEntryPoint',
    description='Entry point for the magres parser.',
    level=1,
    parser_as_interface=False,  # in order to use `child_archives` and auto workflows
    # mainfile_contents_re=r'\$magres-abinitio-v(\d\.)+',
    mainfile_name_re=r'.*\.magres',
)
