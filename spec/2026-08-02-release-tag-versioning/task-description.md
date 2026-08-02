# Task

Make PyPI release builds derive both plugin package versions automatically from the release tag so release commits no longer need to update `plugin.yaml` manually. Normalize release input so `X.Y.Z` and `vX.Y.Z` resolve to one canonical `vX.Y.Z` tag and never silently turn an already-prefixed version into `vvX.Y.Z`. The release is initiated through SourceTree's GitFlow extension.
