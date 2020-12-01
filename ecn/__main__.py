import kblocks.configs  # pylint: disable=unused-import
from absl import app
from kblocks import cli

import ecn.configs  # pylint: disable=unused-import


def cli_main(argv):
    cli.summary_main(cli.get_gin_summary(argv))


if __name__ == "__main__":
    app.run(cli_main)
