function spacetime_error_normalized(learned,truth)
    #err = sum(learned' * truth)/sum(truth' * truth)
    dim1 = size(learned,1)
    dim2 = size(learned,2)
    N = dim1*dim2
    rms_err = sqrt(sum(sum((truth[:,end] - learned[:,end]).^2))/N)
    return rms_err
end