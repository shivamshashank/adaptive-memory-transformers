"""Allow ``python -m amt`` to invoke the command-line interface."""

from amt.cli import main

raise SystemExit(main())
