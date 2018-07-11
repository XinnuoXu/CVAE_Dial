Computing significance using bootstrap
======================================

Prepare:
- Put your 2 or more experiments into different directories (e.g. `A`, `B`, `C`).
- In each directory, the file with system outputs must have the same name (e.g. output.txt, so you have `A/output.txt`, `B/output.txt`, `C/output.txt`).
* You need a context and reference/ground truth file (e.g. context.txt, reference.txt). These are the same for all the experiments.
* You need an (empty) directory for logging and temporary files where you will run the experiment (e.g. `bootstrap-tmp`).
* Set a confidence level, say 95%.

Now run:
```
./compute_bootstrap.py --level 95 bootstrap-tmp context.txt reference.txt output.txt A B C
```
This will compute bootstrap significance tests for every possible pair of your systems (here A-B, A-C, B-C).

Look on the console for outputs like this (or use `2> /dev/null` and it's the only thing that remains, but you won't see progress):
```
A vs B BLEU: System 1 BLEU better: 1000 / 1000 = 1 -- BLEU SIGNIFICANT at 0.95 level
B vs C BLEU: System 2 BLEU better: 723 / 1000 = 0.723
```
It will tell you the p-value (here: 1 and 0.723) and the significance if it's above the confidence level. The first here is significant but the 2nd is not.
