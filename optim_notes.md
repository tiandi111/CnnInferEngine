Optimization Notes
---
1. Operator fusion
    - CONV + BN
    - CONV + RELU
    - BN + RELU
    - 2 CONV SUM -> in-place
    
2. Memory format
    - Use MKL automatically choosed one
    - Joint optimize (graph-level + operator-level)
    
3. In-place
    - BN
    
4. Conv algo
    - Winograd 
        - fast for small filters
    - FFT
        - fast for large filters
    - Strassen’s algorithm
        - O( N³) to O( N².807)
        - less numerical stability
    
    