class ChromehoundsError(Exception):
    """Base error for expected importer failures."""


class FormatError(ChromehoundsError):
    """Raised when an asset is malformed or uses an unsupported variant."""


class AssetNotFoundError(ChromehoundsError):
    """Raised when an asset path is absent from the selected source."""
