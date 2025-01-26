# faststats
Easy to use numba-based speed up of classic numpy statistics methods. 


# TODO: 
# - Define a given max dimension for speedups that is precompiled. 
# - Anything up to this point should be automatically included in import faststats as fs
#   - Anything past that point can be included by demand (if requested with an inplace import or whatever that's called)
# - Any trailing dimensions should always be reshaped to be flat (benchmark this to check if it's worth it...)