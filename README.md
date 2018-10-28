# COMP5328 - Advanced Machine Learning

## Assignment 2 - Learning with Noisy Data

**Author** : Yutong Cao,  Chen Chen,  Yixiong Fang

**Lecturer**: Tongliang Liu

**Tutors** : Nicholas James, Zhuozhuo Tu, Liu Liu

**Objectives:**

Please write objective in two scentences. Do not copy.

**Reuirements:**
- sklearn >= 0.20.0
- multiprocessing
- numpy
- matplotlib


Running Environment Setup
------------


1. Make a folder with name "data" in current directory. Then copy ?? and ?? dataset inside. Please make sure you have the following file tree structure:
     |--- data \\
     ?	|--- ??\\
     ?	|--- ??\\
     |--- nmf \\
     	|--- ??.py \\
      	|--- algorithm.py  \\
      	|--- ??.py  \\
      	|--- main.py \\
      	|--- ??.py \\
      	|--- ??.py \\
      	|--- ??.py \\
      	|--- ??.py \\
      |--- setup.py \\
      |--- README.md \\
3. Run `main.py` in `??` with appropriate dataset name.

   To run `???` on **??** dataset, please run:

   ```
   python nmf/main.py ??
   ```

   To run `???` on **???** dataset, please run:

   ```
   python nmf/main.py ??
   ```
TODO.
All results will be auto-saved to ??? `results/{generated-time-dataname}`. Note that we set the epoch to be 1 in `main.py`. This is because we have 2 (dataset) x 3 (algorithm) = 6 combination in each epoch. This will cost around 4.5 minutes on a i7-6th gen laptop with ORL dataset. However, we increased the epochs to calculate average accuracy and confidence interval etc.