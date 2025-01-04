from scimate_agent.plugins import Plugin, register_plugin


def generate_directory_tree(file_path: str, max_depth: int = 5) -> str:
    ...


@register_plugin
class InspectDatset(Plugin):
    def __call__(
        self,
        file_path: str,
        max_depth: int = 5,
    ) -> str:
        directory_tree = generate_directory_tree(file_path, max_depth)
        return directory_tree
