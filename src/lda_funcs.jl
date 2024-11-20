#=
The functions required for the Length Difference and Angle Difference (LDA) loss function 
=#

normed_ld(a,b) = abs(norm(a)-norm(b))/(norm(a)+norm(b))
cosine_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
cosine_distance(a, b) = (1 - cosine_similarity(a, b))/2