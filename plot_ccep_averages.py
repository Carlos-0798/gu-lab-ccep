"""Show Mouse 1 LP/SC using the supplied CCEP bandpass and native plot style.

Default: 0.5-30.5 Hz bandpass. Use --mode no_extra_filter for the earlier
baseline-only result, or --compare for a side-by-side filter comparison.
"""

import sys

from filter_ccep import main

if __name__ == "__main__":
    main(["--mouse", "1", *sys.argv[1:]])
