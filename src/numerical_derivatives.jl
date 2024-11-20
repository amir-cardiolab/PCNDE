#=
Here I define the numerical derivatives used in both the true partial differential equation (PDE) 
and the neural partial differential equation (NPDE). The true PDE is used to generate the training data. 
The NPDE is trained off of the true PDE generated training data on lower finite difference accuracy. 
=#

############# First derivative
# First Order Forward
function  f1_firstOrder_forward(n,dx)
    ∂x1 = (diagm(
                0 => -1 * ones(n),
                1 =>  1 * ones(n-1)
         )) ./ dx
    # periodic boundaries
    ∂x1[1,end] = 0 
    ∂x1[end,1] = 1 / dx 
    
    # sparsify
    ∂x1 = sparse(∂x1)

    return ∂x1
end

# First Order Backward
function  f1_firstOrder_backward(n,dx)
    ∂x1 = (diagm(
                0 =>  1 * ones(n),
                -1 =>  -1 * ones(n-1)
         )) ./ dx
    # periodic boundaries
    ∂x1[1,end] = -1 / dx 
    ∂x1[end,1] = 0

    # sparsify
    ∂x1 = sparse(∂x1)
    
    return ∂x1
end

# Second Order Central
function f1_secondOrder_central(n,dx)
    ∂x1 = (diagm(
                -1 => -1*ones(n-1),
                 0 => zeros(n),
                 1 => ones(n-1) 
                )) ./ (2*dx)
    # periodic boundaries
    ∂x1[1,end] = -1 / (2*dx)
    ∂x1[end,1] = 1 / (2*dx)

    # sparsify
    ∂x1 = sparse(∂x1)

    return ∂x1
end

# Second Order Forward
function f1_secondOrder_forward(n,dx)
    ∂x1 = (diagm(
                 0 => -3 * ones(n),
                 1 => 4 * ones(n-1),
                 2 => -1 * ones(n-2), 
                )) ./ (2*dx)
    # periodic boundaries
    ∂x1[1,end] = 0
    ∂x1[end,1] = 4 / (2*dx)
    ∂x1[end-1,1] = -1 / (2*dx)
    ∂x1[end,2] = -1 / (2*dx)

    # sparsify
    ∂x1 = sparse(∂x1)

    return ∂x1
end

# Second Order Backward
function f1_secondOrder_backward(n,dx)
    ∂x1 = (diagm(
                 0 => 3 * ones(n),
                 -1 => -4 * ones(n-1),
                 -2 => 1 * ones(n-2), 
                )) ./ (2*dx)
    # periodic boundaries
    ∂x1[end,1] = 0
    ∂x1[1,end] = -4 / (2*dx)
    ∂x1[1,end-1] = 1 / (2*dx)
    ∂x1[2,end] = 1 / (2*dx)

    # sparsify
    ∂x1 = sparse(∂x1)

    return ∂x1
end

# Fourth Order Central
function f1_fourthOrder_central(n,dx)
    ∂x1 = (diagm(
                 0 => zeros(n),
                 -1 => -8 * ones(n-1),
                 1 => 8 * ones(n-1),
                 -2 => 1 * ones(n-2), 
                 2 => -1 * ones(n-2)
                )) ./ (12*dx)
    # periodic boundaries
    ∂x1[1,end] = -8 / (12*dx)
    ∂x1[1,end-1] = 1 / (12*dx)
    ∂x1[2,end] = 1 / (12*dx)
    ∂x1[end,1] = 8 / (12*dx)
    ∂x1[end,2] = -1 / (12*dx)
    ∂x1[end-1,1] = -1 / (12*dx)

    # sparsify
    #∂x1 = sparse(∂x1)

    return ∂x1
end

# Sixth Order Central
function f1_sixthOrder_central(n,dx)
    ∂x1 = (diagm(
                 0 => zeros(n),
                 -1 => -45 * ones(n-1),
                 1 => 45 * ones(n-1),
                 -2 => 9 * ones(n-2), 
                 2 => -9 * ones(n-2),
                 -3 => -1 * ones(n-3),
                  3 =>  1 * ones(n-3),
                 )) ./ (60*dx)
    # periodic boundaries
    ∂x1[1,end] = -45 / (60*dx)
    ∂x1[1,end-1] = 9 / (60*dx)
    ∂x1[1,end-2] = -1 / (60*dx)
    ∂x1[2,end] = 9 / (60*dx)
    ∂x1[2,end-1] = -1 / (60*dx)
    ∂x1[3,end] = -1 / (60*dx)

    ∂x1[end,1] = 45 / (60*dx)
    ∂x1[end,2] = -9 / (60*dx)
    ∂x1[end,3] = 1 / (60*dx)
    ∂x1[end-1,1] = -9 / (60*dx)
    ∂x1[end-1,2] = 1 / (60*dx)    
    ∂x1[end-2,1] = 1 / (60*dx)

    # sparsify
    #∂x1 = sparse(∂x1)

    return ∂x1
end

# Eigth Order Central
function f1_eigthOrder_central(n,dx)
    ∂x1 = (diagm(
                 0 => zeros(n),
                 -1 => -672 * ones(n-1),
                 1 => 672 * ones(n-1),
                 -2 => 168 * ones(n-2), 
                 2 => -168 * ones(n-2),
                 -3 => -32 * ones(n-3),
                  3 =>  32 * ones(n-3),
                 -4 => 3 * ones(n-4),
                  4 => -3 * ones(n-4)
                 )) ./ (840*dx)
    # periodic boundaries
    ∂x1[1,end] = -672 / (840*dx)
    ∂x1[1,end-1] = 168 / (840*dx)
    ∂x1[1,end-2] = -32 / (840*dx)
    ∂x1[1,end-3] = 3 / (840*dx)
    ∂x1[2,end] = 168 / (840*dx)
    ∂x1[2,end-1] = -32 / (840*dx)
    ∂x1[2,end-2] = 3 / (840*dx)
    ∂x1[3,end] = -32 / (840*dx)
    ∂x1[3,end-1] = 3 / (840*dx)
    ∂x1[4,end] = 3 / (840*dx)


    ∂x1[end,1] = 672 / (840*dx)
    ∂x1[end,2] = -168 / (840*dx)
    ∂x1[end,3] = 32 / (840*dx)
    ∂x1[end,4] = -3 / (840*dx)
    ∂x1[end-1,1] = -168 / (840*dx)
    ∂x1[end-1,2] = 32 / (840*dx)    
    ∂x1[end-1,3] = -3 / (840*dx)
    ∂x1[end-2,1] = 32 / (840*dx)    
    ∂x1[end-2,2] = -3 / (840*dx)
    ∂x1[end-3,1] = -3 / (840*dx)

    # sparsify
    #∂x1 = sparse(∂x1)

    return ∂x1
end


############# Second derivative

# First Order Forward
function f2_firstOrder_forward(n,dx)
    ∂x2 = (diagm(
                0 => 1 * ones(n),
                1 => -2 * ones(n-1),
                2 => 1 * ones(n-2)
                )) ./ (dx^2)
    # periodic boundaries
    ∂x2[end,1] = -2 / (dx^2)
    ∂x2[end,2] = 1 / (dx^2)
    ∂x2[end-1,1] = 1 / (dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)

    return ∂x2
end


# First Order Backward
function f2_firstOrder_backward(n,dx)
    ∂x2 = (diagm(
                0 => 1 * ones(n),
                -1 => -2 * ones(n-1),
                -2 => 1 * ones(n-2)
                )) ./ (dx^2) 
    # periodic boundaries
    ∂x2[1,end] = -2 / (dx^2)
    ∂x2[1,end-1] = 1 / (dx^2)
    ∂x2[2,end] = 1 / (dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end

# Second Order Forward
function f2_secondOrder_forward(n,dx)
    ∂x2 = (diagm(
                0 => 2 * ones(n),
                1 => -5 * ones(n-1),
                2 => 4 * ones(n-2),
                3 => -1 * ones(n-3)
                )) ./ (dx^2)
    # periodic boundaries
    ∂x2[end,1] = -5 / (dx^2)
    ∂x2[end,2] = 4 / (dx^2)
    ∂x2[end,3] = -1 / (dx^2)
    ∂x2[end-1,1] = 4 / (dx^2)
    ∂x2[end-1,2] = -1 / (dx^2)
    ∂x2[end-2,1] = -1 / (dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end

# Second Order Backward
function f2_secondOrder_backward(n,dx)
    ∂x2 = (diagm(
                0 => 2 * ones(n),
                -1 => -5 * ones(n-1),
                -2 => 4 * ones(n-2),
                -3 => -1 * ones(n-3)
                )) ./ (dx^2)
    # periodic boundaries
    ∂x2[1,end] = -5 / (dx^2)
    ∂x2[1,end-1] = 4 / (dx^2)
    ∂x2[1,end-2] = -1 / (dx^2)
    ∂x2[2,end] = 4 / (dx^2)
    ∂x2[2,end-1] = -1 / (dx^2)
    ∂x2[3,end] = -1 / (dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end

# Second Order Central
function f2_secondOrder_central(n,dx)
    ∂x2 = (diagm(
                0 => -2 * ones(n),
                -1 => 1 * ones(n-1),
                 1 => 1 * ones(n-1)
                )) ./ (dx^2)
    # periodic boundaries
    ∂x2[1,end] = 1 / (dx^2)
    ∂x2[end,1] = 1 / (dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end

# Fourth Order central
function f2_fourthOrder_central(n,dx)
    ∂x2 = (diagm(
                0 => -30 * ones(n),
                -1 => 16 * ones(n-1),
                1 => 16 * ones(n-1),
                -2 => -1 * ones(n-2),
                2 => 1 * ones(n-2)
                )) ./ (12 * dx^2)
    # periodic boundaries
    ∂x2[1,end] = 16 / (12 * dx^2)
    ∂x2[1,end-1] = -1 / (12 * dx^2)
    ∂x2[2,end] = -1 / (12 * dx^2)
    ∂x2[end,1] = 16 / (12 * dx^2)
    ∂x2[end,2] = -1 / (12 * dx^2)
    ∂x2[end-1,1] = -1 / (12 * dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end

# Sixth Order central
function f2_sixthOrder_central(n,dx)
    ∂x2 = (diagm(
                0 => -490 * ones(n),
                -1 => 270 * ones(n-1),
                1 => 270 * ones(n-1),
                -2 => -27 * ones(n-2),
                2 =>  -27 * ones(n-2),
                -3 =>  2 * ones(n-3),
                3 =>  2 * ones(n-3),
                )) ./ (180 * dx^2)
    # periodic boundaries
    ∂x2[1,end] = 270 / (180 * dx^2)
    ∂x2[1,end-1] = -27 / (180 * dx^2)
    ∂x2[1,end-2] = 2 / (180 * dx^2)
    ∂x2[2,end] = -27 / (180 * dx^2)
    ∂x2[2,end-2] = 2 / (180 * dx^2)
    ∂x2[3,end] = 2 / (180 * dx^2)

    ∂x2[end,1] = 270 / (180 * dx^2)
    ∂x2[end,2] = -27 / (180 * dx^2)
    ∂x2[end,3] = 2 / (180 * dx^2)
    ∂x2[end-1,1] = -27 / (180 * dx^2)
    ∂x2[end-1,2] = 2 / (180 * dx^2)
    ∂x2[end-2,1] = 2 / (180 * dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end

# Eigth Order central
function f2_eigthOrder_central(n,dx)
    ∂x2 = (diagm(
                0 => -14350 * ones(n),
                -1 => 8064 * ones(n-1),
                1 => 8064 * ones(n-1),
                -2 => -1008 * ones(n-2),
                2 => -1008 * ones(n-2),
                -3 => 128 * ones(n-3),
                3 => 128 * ones(n-3),
                -4 => -9 * ones(n-4),
                4 => -9 * ones(n-4)
                )) ./ (5040 * dx^2)
    # periodic boundaries
    ∂x2[1,end] = 8064 / (5040 * dx^2)
    ∂x2[1,end-1] = -1008 / (5040 * dx^2)
    ∂x2[1,end-2] = 128 / (5040 * dx^2)
    ∂x2[1,end-3] = -9 / (5040 * dx^2)
    ∂x2[2,end] = -1008 / (5040 * dx^2)
    ∂x2[2,end-1] = 128 / (5040 * dx^2)
    ∂x2[2,end-2] = -9 / (5040 * dx^2)
    ∂x2[3,end] = 128 / (5040 * dx^2)
    ∂x2[3,end-1] = -9 / (5040 * dx^2)
    ∂x2[4,end] = -9 / (5040 * dx^2)

    ∂x2[end,1] = 8064 / (5040 * dx^2)
    ∂x2[end,2] = -1008 / (5040 * dx^2)
    ∂x2[end,3] = 128 / (5040 * dx^2)
    ∂x2[end,4] = -9 / (5040 * dx^2)
    ∂x2[end-1,1] = -1008 / (5040 * dx^2)
    ∂x2[end-1,2] = 128 / (5040 * dx^2)
    ∂x2[end-1,3] = -9 / (5040 * dx^2)
    ∂x2[end-2,1] = 128 / (5040 * dx^2)
    ∂x2[end-2,2] = -9 / (5040 * dx^2)
    ∂x2[end-3,1] = -9 / (5040 * dx^2)

    # sparsify
    ∂x2 = sparse(∂x2)    

    return ∂x2
end



############# Third derivative

# Second Order Central
function f3_secondOrder_central(n,dx)
    ∂x3 = (diagm(
                0 => zeros(n),
                -1 => 2 * ones(n-1),
                1 => -2 * ones(n-1),
                -2 => -1 * ones(n-2),
                2 => 1 * ones(n-2)
                )) ./ (2 * dx^3)
    # periodic boundaries
    ∂x3[1,end] = 2 / (2 * dx^3)
    ∂x3[1,end-1] = -1 / (2 * dx^3)
    ∂x3[2,end] = -1 / (2 * dx^3)
    ∂x3[end,1] = -2 / (2 * dx^3)
    ∂x3[end,2] = 1 / (2 * dx^3)
    ∂x3[end-1,1] = 1 / (2 * dx^3)

    # sparsify
    ∂x3 = sparse(∂x3)    

    return ∂x3
end

# Fourth Order Central
function f3_fourthOrder_central(n,dx)
    ∂x3 = (diagm(
                0 => zeros(n),
                -1 => 13 * ones(n-1),
                1 => -13 * ones(n-1),
                -2 => -8 * ones(n-2),
                2 => 8 * ones(n-2),
                -3 => 1 * ones(n-3),
                3 => -1 * ones(n-3)
                )) ./ (8 * dx^3)
    # periodic boundaries
    ∂x3[1,end] = 13 / (8 * dx^3)
    ∂x3[1,end-1] = -8 / (8 * dx^3)
    ∂x3[1,end-2] = 1 / (8 * dx^3)
    ∂x3[2,end] = -8 / (8 * dx^3)
    ∂x3[2,end-1] = 1 / (8 * dx^3)
    ∂x3[3,end] = 1 / (8 * dx^3)
    ∂x3[end,1] = -13 / (8 * dx^3)
    ∂x3[end,2] = 8 / (8 * dx^3)
    ∂x3[end,3] = -1 / (8 * dx^3)
    ∂x3[end-1,1] = 8 / (8 * dx^3)
    ∂x3[end-1,2] = -1 / (8 * dx^3)
    ∂x3[end-2,1] = 1 / (8 * dx^3)

    # sparsify
    ∂x3 = sparse(∂x3)     

    return ∂x3
end

# Sixth Order Central
function f3_sixthOrder_central(n,dx)
    ∂x3 = (diagm(
                0 => zeros(n),
                -1 => 488 * ones(n-1),
                1 => -488 * ones(n-1),
                -2 => -338 * ones(n-2),
                2 => 338 * ones(n-2),
                -3 => 72 * ones(n-3),
                3 => -72 * ones(n-3),
                -4 => -7 * ones(n-4),
                4 => 7 * ones(n-4)
                )) ./ (240 * dx^3)
    # periodic boundaries
    ∂x3[1,end] = 488 / (240 * dx^3)
    ∂x3[1,end-1] = -338 / (240 * dx^3)
    ∂x3[1,end-2] = 72 / (240 * dx^3)
    ∂x3[1,end-3] = -7 / (240 * dx^3)
    ∂x3[2,end] = -338 / (240 * dx^3)
    ∂x3[2,end-1] = 72 / (240 * dx^3)
    ∂x3[2,end-2] = -7 / (240 * dx^3)
    ∂x3[3,end] = 72 / (240 * dx^3)
    ∂x3[3,end-1] = -7 / (240 * dx^3)
    ∂x3[4,end] = -7 / (240 * dx^3)

    ∂x3[end,1] = -488 / (240 * dx^3)
    ∂x3[end,2] = 338 / (240 * dx^3)
    ∂x3[end,3] = -72 / (240 * dx^3)
    ∂x3[end,4] = 7 / (240 * dx^3)
    ∂x3[end-1,1] = 338 / (240 * dx^3)
    ∂x3[end-1,2] = -72 / (240 * dx^3)
    ∂x3[end-1,3] = 7 / (240 * dx^3)
    ∂x3[end-2,1] = -72 / (240 * dx^3)
    ∂x3[end-2,2] = 7 / (240 * dx^3)
    ∂x3[end-3,1] = 7 / (240 * dx^3)

    # sparsify
    ∂x3 = sparse(∂x3)     

    return ∂x3
end

############# Fourth derivative

# Second Order Central
function f4_secondOrder_central(n,dx)
    ∂x4 = (diagm(
                0 => 6 * ones(n),
                -1 => -4 * ones(n-1),
                 1 => -4 * ones(n-1),
                 -2 => 1 * ones(n-2),
                  2 => 1 * ones(n-2)
                )) ./ (dx^4)
    # periodic boundaries
    ∂x4[1,end] = -4 / (dx^4)
    ∂x4[1,end-1] = 1 / (dx^4)
    ∂x4[2,end] =  1 / (dx^4)

    ∂x4[end,1] = -4 / (dx^4)
    ∂x4[end,2] = 1 / (dx^4)
    ∂x4[end-1,1] = 1 / (dx^4)

    # sparsify
    ∂x4 = sparse(∂x4)     

    return ∂x4
end

# Fourth Order Central
function f4_fourthOrder_central(n,dx)
    ∂x4 = (diagm(
                0 => 56 * ones(n),
                -1 => -39 * ones(n-1),
                1 => -39 * ones(n-1),
                -2 => 12 * ones(n-2),
                2 => 12 * ones(n-2),
                -3 => -1 * ones(n-3),
                3 => -1 * ones(n-3)
                )) ./ (6 * dx^4)
    # periodic boundaries
    ∂x4[1,end] = -39 / (6 * dx^4)
    ∂x4[1,end-1] = 12 / (6 * dx^4)
    ∂x4[1,end-2] = -1 / (6 * dx^4)
    ∂x4[2,end] = 12 / (6 * dx^4)
    ∂x4[2,end-1] = -1 / (6 * dx^4)
    ∂x4[3,end] = -1 / (6 * dx^4)
    ∂x4[end,1] = -39 / (6 * dx^4)
    ∂x4[end,2] = 12 / (6 * dx^4)
    ∂x4[end,3] = -1 / (6 * dx^4)
    ∂x4[end-1,1] = 12 / (6 * dx^4)
    ∂x4[end-1,2] = -1 / (6 * dx^4)
    ∂x4[end-2,1] = -1 / (6 * dx^4)

    # sparsify
    ∂x4 = sparse(∂x4)     

    return ∂x4
end


# Sixth Order Central
function f4_sixthOrder_central(n,dx)
    ∂x4 = (diagm(
                0 => 2730 * ones(n),
                -1 => -1952 * ones(n-1),
                 1 => -1952 * ones(n-1),
                -2 => 676 * ones(n-2),
                 2 => 676 * ones(n-2),
                -3 => -96 * ones(n-3), 
                 3 => -96 * ones(n-3), 
                -4 => 7 * ones(n-4),
                 4 => 7 * ones(n-4)
                )) ./ (240 * dx^4)
    # periodic boundaries
    ∂x4[1,end] = -1952 / (240 * dx^4)
    ∂x4[1,end-1] = 676 / (240 * dx^4)
    ∂x4[1,end-2] = -96 / (240 * dx^4)
    ∂x4[1,end-3] = 7 / (240 * dx^4)
    ∂x4[2,end] = 676 / (240 * dx^4)
    ∂x4[2,end-1] = -96 / (240 * dx^4)
    ∂x4[2,end-2] = 7 / (240 * dx^4)
    ∂x4[3,end] = -96 / (240 * dx^4)
    ∂x4[3,end-1] = 7 / (240 * dx^4)
    ∂x4[4,end] = 7 / (240 * dx^4)
    ∂x4[end,1] = -1952 / (240 * dx^4)
    ∂x4[end,2] = 676 / (240 * dx^4)
    ∂x4[end,3] = -96 / (240 * dx^4)
    ∂x4[end,4] =  7 / (240 * dx^4)
    ∂x4[end-1,1] = 676 / (240 * dx^4)
    ∂x4[end-1,2] = -96 / (240 * dx^4)
    ∂x4[end-1,3] =  7 / (240 * dx^4)
    ∂x4[end-2,1] = -96 / (240 * dx^4)
    ∂x4[end-2,2] =  7 / (240 * dx^4)
    ∂x4[end-3,1] =  7 / (240 * dx^4)

    # sparsify
    ∂x4 = sparse(∂x4)      

    return ∂x4
end