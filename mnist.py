# August Leander Hedman
# augu1789@edu.nextkbh.dk
# NEXT Sukkertoppen, S3n

import numpy as np
import gzip
import struct


        # Kald backprop for  et layer[i-1] med et argument værdier[i-1]
        # Definer et array som gemmer alle aktiveringsværdierne for hvert lag for et givent træningseksempel. Find således disse værdier
        #o utput error er givet ved a[:-1] - one_hot(Label)
        # # Vi ønsker at finde alle aktiveringsværdier og alle z-værdier gennem hele netværket for et givent træningseksempel