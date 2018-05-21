import sys

def exc_info_hook(exc_type, value, tb):
    """An exception hook that starts IPdb automatically on error if in interactive mode."""

    if hasattr(sys, 'ps1') or not sys.stderr.isatty() or exc_type == KeyboardInterrupt:
        # we are in interactive mode, we don't have a tty-like
        # device,, or the user triggered a KeyboardInterrupt,
        # so we call the default hook
        sys.__excepthook__(exc_type, value, tb)
    else:
        import traceback
        # import ipdb
        import pudb
        # we are NOT in interactive mode, print the exception
        traceback.print_exception(exc_type, value, tb)
        print
        raw_input("Press any key to start debugging...")
        # then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        # ipdb.post_mortem(tb)  # more modern
        pudb.post_mortem(tb)

