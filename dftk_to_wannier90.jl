using ProgressMeter
using LinearAlgebra
using Dates



##!!!!!!!!!! !!!! !  !             read file              !  ! !!!! !!!!!!!!!!##

"""
Read the .nnkp file provided by the preprocessing routine of Wannier90 (i.e. "wannier90.x -pp prefix")
Returns:
1) the array of k points, they respective neirest neighbours and associated shifing vectors (non zero if the neighbour of is located in another cell).
2) the array of projections. Each line contains the informations for one projection :  [center],[quantum numbers: l,mr,r],[ref. z_axis],[ref. x_axis],α = Z/a
"""
function read_nnkp_file(prefix::String)
    ## Read file
    if !isfile("$prefix.nnkp")
        error("file $prefix.nnkp not found.")    # Check the presence of the file
    end
    
    file = open("$prefix.nnkp")
    @info "Reading nnkp file", file 
    ln = [string(l) for l in eachline(file)]
    close(file)


    ## Extract nnkp block 
    i_nn_kpts = findall(x-> endswith(x,"nnkpts"),ln)                               #Indices of the first and last line of the nnkpts block
    @assert size(i_nn_kpts,1) == 2
    char_to_vec(line) = [parse.(Int,x) for x in split(line,' ',keepempty=false)]
    nn_kpts =  [ char_to_vec(l) for l in ln[i_nn_kpts[1]+2:i_nn_kpts[2]-1] ]



    ## Extract projections block
    i_projs = findall(x-> endswith(x,"projections"),ln)                                  #Indices of the first and last line of the projection block
    @assert size(i_projs,1) == 2
    raw_projs = [split(ln[i],' ',keepempty = false) for i in i_projs[1]+1:i_projs[2]-1]  #data in strings

    # reshape so that one line gives all infos about one projection
    n_projs = parse(Int,only(popfirst!(raw_projs)))
    @assert(n_projs == size(raw_projs,1)/2)
    raw_projs = reshape(raw_projs,(2,n_projs))

    # Parse in the format proj = [ [[center],[l,mr,r],[z_axis],[x_axis], α], ... ]
    projs = []
    for j in 1:n_projs
        center = [parse(Float64,x) for x in raw_projs[1,j][1:3]]
        quantum_numbers = [parse(Int,x) for x in raw_projs[1,j][4:end] ]
        z_axis = [parse(Float64,x) for x in raw_projs[2,j][1:3]]
        x_axis = [parse(Float64,x) for x in raw_projs[2,j][4:6]]
        α = parse(Float64,raw_projs[2,j][7])
        push!(projs, [center,quantum_numbers,z_axis,x_axis,α])
    end
    
    nn_kpts,projs
    
end




##!!!!!!!!!! !!!! !  !              eig file              !  ! !!!! !!!!!!!!!!##

"""
Take scf_res, the result of a self_consistent_field calculation and writes eigenvalues in a format readable by Wannier90.
"""

function generate_eig_file(prefix::String,scf_res)
    #energies have to be in EV
    Ha_to_Ev = 27.2114

    eigvalues = scf_res.eigenvalues
    eigvalues .*= Ha_to_Ev
    k_size = size(eigvalues,1)
    n_bands = size(eigvalues[1],1)

    #write file
    open("$prefix.eig","w") do f
        for k in 1:k_size
            for n in 1:n_bands
                write(f,"  $n  $k  $(eigvalues[k][n])"*"\n")
            end
        end
    end

    nothing
    
end



##!!!!!!!!!! !!!! !  !              mmn file              !  ! !!!! !!!!!!!!!!##

"""
 Computes the matrix [M^{k,b}]_{m,n} = <u_{m,k} | u_{n,k+b}> for given k, kpb = k+b.
 
 Returns in the format of a (n_bands^2, 2) matrix, where each line is one overlap, and columns are real and imaginary parts.

 K_shift is the "shifting" vector correction due to the periodicity conditions imposed on k -> ψ_k.
 We use here that : u_{n(k + K_shift)}(r) = e^{-i*<K_shift,r>} u_{nk}
"""
function get_overlap(k,kpb,n_bands,ψ,pw_basis::PlaneWaveBasis; K_shift = [0,0,0])
    Mkb = zeros(Float64,n_bands*n_bands,2)
    current_line = 0 #Manual count necessary to match the format of .mmn files.

    for n in 1:n_bands
        for m in 1:n_bands
            
            ovlp = 0im
            
            #Extract Fourier coeffs and corresponding vectors in reciprocal lattice
            Gk_coeffs = ψ[k][:,m] 
            Gk_vec = G_vectors(pw_basis.kpoints[k])
            Gkpb_coeffs = ψ[kpb][:,n]
            Gkpb_vec = [ G - K_shift for G in G_vectors(pw_basis.kpoints[kpb]) ] # Don't forget the shift, see the DOC block
            
            #Search for common Fourier modes, corresponding indices are written in map_fourier_mode
            map_fourier_modes = []
            for G1 in Gk_vec
                for G2 in Gkpb_vec
                    if  G1 == G2
                        iG1 = only(findall(x-> x==G1,Gk_vec))
                        iG2 = only(findall(x-> x==G2,Gkpb_vec))
                        push!(map_fourier_modes,[iG1,iG2])
                    end
                end
            end
            
            #Compute the overlap for mn
            for (i,j) in map_fourier_modes
                ovlp += conj(Gk_coeffs[i])*Gkpb_coeffs[j]
            end
            
            current_line += 1 #Go to the next line
            Mkb[current_line,:] = [real(ovlp),imag(ovlp)]

        end
    end

    Mkb

 end



"""
Iteratively use the preceeding function on each k and kpb to generate the whole .mmn file.

"""
function generate_mmn_file(prefix::String,ψ,pw_basis::PlaneWaveBasis, nn_kpts)
    #general parameters
    n_bands = size(ψ[1],2)
    k_size = length(ψ)
 
    progress = Progress(only(size(nn_kpts)),desc = "Computing Mmn overlaps : ")
    
    #Small function for the sake of clarity
    read_nn_kpts(n) = nn_kpts[n][1],nn_kpts[n][2],nn_kpts[n][3:end]

    #Write file
    open("$prefix.mmn","w") do f
        write(f,"Generated by DFTK at ",string(now()),"\n")
        write(f,string(n_bands)*"   "*string(k_size)*"   "*string(n_bands)*"\n") #TODO num_wan
        
        for i_nnkp in 1:only(size(nn_kpts)) #Loop over all (k_points, neirest_neighbour, shif_vector)
            #Label of the matrix
            k,nnk,shift = read_nn_kpts(i_nnkp)
            write(f,string(k)*"  "*string(nnk)*"  "*string(shift[1])*"  "*string(shift[2])*"  "*string(shift[3])*"\n")   
            #Overlaps
            Mkb = get_overlap(k,nnk,n_bands,ψ,pw_basis; K_shift = shift)
            for i in 1:n_bands*n_bands
                write(f, string(Mkb[i,1])*" "*string(Mkb[i,2])*"\n")
            end
            next!(progress)
        end
    end

end



##!!!!!!!!!! !!!! !  !              amn file              !  ! !!!! !!!!!!!!!!##

"""
Given a k points, provide the indices of the corresponding G_vectors in the general FFT grid.
"""
function index_fourier_modes(pw_basis::PlaneWaveBasis,k,fft_grid)
    map = []
    for G in G_vectors(pw_basis.kpoints[k])
        iG = only( findall(x -> x==G, fft_grid) )
        push!(map,iG)
    end
    map
    
end


"""
Compute matrix [A_k]_{m,n} = < ψ_m^k | g_per_n > where g_per_n are periodized gaussians whose respective centers are given as an (n_bands,1) array [ [center 1], ... ].

The dot product is computed in the Fourier space. 

Given a gaussian g_n, the periodized gaussian is defined by : g_per_n =  ∑_{R in lattice} g_n( ⋅ - R). 

The fourier coefficients of g_per_n at any G is given by the value of the fourier transform of g_n at G.
"""
function compute_amn_k_gaussians(pw_basis::PlaneWaveBasis,ψ,k,centers)

    ## Before calculation 
    guess_fourier(center) = xi ->  exp(-im*dot(xi,center) - dot(xi,xi)/4)  #associate a center with the fourier transform of the corresponding gaussian
    n_bands = size(ψ[1][1,:],1)
    n_guess = size(centers,1)

    fft_grid = [G for (iG,G) in enumerate(G_vectors(pw_basis)) ]    #FFT grid in recip lattice coordinates
    G_cart =[ pw_basis.model.recip_lattice * G for G in fft_grid ]  #FFT grid in cartesian coordinates
    index = index_fourier_modes(pw_basis,k,fft_grid)                #Indices of the Fourier modes of the bloch states in the whole FFT_grid for given k

    #Initialize output
    A_k = zeros(Complex,(n_bands,n_guess))

    ## Compute A_k
    for n in 1:n_guess

        ## compute fourier coeffs of g_per_n
        fourier_gn = guess_fourier(centers[n])
        norm_g_per_n = norm([fourier_gn(G) for G in G_cart],2)                          # functions are l^2 normalized in Fourier, in DFTK conventions.
        coeffs_g_per_n = [ fourier_gn(G_cart[iG]) for iG in index ]  ./ norm_g_per_n    # Fourier coeffs of gn for G_vectors in common with ψm
        
        for m in 1:n_bands
            coeffs_ψm = ψ[k][:,m]
            A_k[m,n] = dot(coeffs_ψm,coeffs_g_per_n)                                    #The first argument is conjugated with the Julia "dot" function
        end
        
    end

    A_k
end


"""
Use iteratively the preceding function to generate the .amn file 
"""
function generate_amn_file(prefix::String,ψ,pw_basis::PlaneWaveBasis, projs)
    # general parameters
    n_bands = size(ψ[1],2)
    k_size = length(ψ)

    progress = Progress(k_size,desc = "Computing Amn overlaps : ")

    #centers of the gaussian guesses for silicon
    centers = [[-0.125,-0.125, 0.375], [0.375,-0.125,-0.125], [-0.125, 0.375,-0.125], [-0.125,-0.125,-0.125]]

    ## write file
    open("$prefix.amn","w") do f
        write(f,"Generated by DFTK at ",string(now()),"\n")
        write(f,string(n_bands)*"   "*string(k_size)*"   "*string(n_bands)*"\n") #TODO num_wan pour le dernier
        for k in 1:k_size
            A_k = compute_amn_k_gaussians(pw_basis,ψ,k,centers)
            for m in 1:size(A_k,1)
                for n in 1:size(A_k,2)
                    write(f,"$m  $n  $k  $(real(A_k[m,n]))  $(imag(A_k[m,n]))"*"\n")
                end
            end
            next!(progress)
        end
         
    end         

end




###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                   Work in progress                   !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###



#TODO CREER UNE CLASSE OU STRUCTURE PROJECTIONS QUI CONTIENT CES INFOS
# Etendre à d'autres type de projections

# Associate quantum numbers with sp3 type orbitals  

# #Hydrogene AOS
# s(θ,φ) = 1/√(4*pi)
# pz(θ,φ) = √( 3/(4*π) )*cos(θ)
# px(θ,φ) = √( 3/(4*π) )*sin(θ)*cos(φ)
# py(θ,φ) = √( 3/(4*π) )*sin(θ)*sin(φ) 

# angular_parts = Dict()
# push!(angular_parts, [-3,1] => (θ,φ) -> 0.5*(s(θ,φ)+px(θ,φ)+py(θ,φ)+pz(θ,φ)) )  #sp3-1
# push!(angular_parts, [-3,2] => (θ,φ) -> 0.5*(s(θ,φ)+px(θ,φ)-py(θ,φ)-pz(θ,φ)) )  #sp3-2
# push!(angular_parts, [-3,3] => (θ,φ) -> 0.5*(s(θ,φ)-px(θ,φ)+py(θ,φ)-pz(θ,φ)) )  #sp3-3
# push!(angular_parts, [-3,4] => (θ,φ) -> 0.5*(s(θ,φ)-px(θ,φ)-py(θ,φ)+pz(θ,φ)) )  #sp3-4

# radial_parts = Dict()
# push!(radial_parts, 1 => (r,α) -> 2*α^(3/2)*exp(-α*r) ) 
# push!(radial_parts, 2 => (r,α) -> (1/(2*√2)) * α^(3/2) * (2-α*r) * exp(-α*r/2) )
# push!(radial_parts, 3 => (r,α) -> √(4/27) * α^(3/2) * (1-2*α*r/3+2*(α^2)*(r^2)/27) * exp(-α*r/3) ) 


# function generate_guess(proj,dic_R,dic_Θ)
#     ############### BEGIN DOC
#     #
#     # Produces one guess function : takes  one line of the projections table
#     # given by ~read_nnkp_file~ and  dictionaries linking  quantum numbers
#     # and angular (Θ) or radial parts (R) of hydrogene AOs. 
#     #
#     # Recall that : proj = [ [center], [quantum numbers], [z_axis], [x_axis], α ]
#     #
#     # Ne pas confondre Θ majuscule et θ minuscule...
#     ############### END_DOC

#     center = proj[1]
#     l,mr,r = proj[2]
#     α = proj[5]

#     #TODO take into account non-canonical basis
#     @assert(proj[3] == [0.00,0.00,1.00]) #For now we limit ourselves to the canonical basis
#     @assert(proj[4] == [1.00,0.00,0.00])

#     #Choose radial and angular parts
#     radial_part = get!(dic_R,r,1)
#     angular_part = get!(dic_Θ, [l,mr],1)
    
#     function g_n(r,θ,φ)
#         radial_part(r)*angular_part(θ,φ)
#     end

#     g_n
    
# end



################################################################################
##                             GENERATE ALL FILES                             ##
################################################################################

"""
Use the above functions to read the nnkp file and generate the .eig, .amn and .mmn files needed by Wannier90.
"""
function generate_input_files(prefix::String,scf_res,pw_basis::PlaneWaveBasis;
                              write_amn = true, write_mmn = true, write_eig = true)
    
    ψ = scf_res.ψ #just renamig
    nn_kpts,projs = read_nnkp_file(prefix,ψ) #read the .nnkp file

    ## Generate_files
    if write_eig
        generate_eig_file("Si",scf_res)
    end
    
    if write_amn
        generate_amn_file("Si",ψ,pw_basis,projs)
    end
    
    if write_mmn
        generate_mmn_file("Si",ψ,pw_basis,nn_kpts)
    end
    
    return nothing

end








