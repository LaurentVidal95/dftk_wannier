using DFTK
using ProgressMeter
using LinearAlgebra

include("../dftk_to_wannier90.jl")

########################################################
#                                                      #
##### ### #              SCF                 # ### #####
#                                                      #
########################################################

#TEST FOR SILICON

a = 10.26 #a.u.

# Note that vectors are stored in rows
lattice = a / 2*[[-1.  0. -1.];   #basis.model.lattice (in a.u.)
                 [ 0   1.  1.];
                 [ 1   1.  0.]]

#One atom
Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]] ]


## Tests on several atoms. Random system in order to test routines on the basis.
# Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
# Mg = ElementPsp(:Mg, psp=load_psp("hgh/pbe/Mg-q2"))
# atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]]
#           Mg => [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]]
#           ]


model = model_PBE(lattice,atoms)


kgrid = [4,4,4] # mp grid
Ecut = 20.0

#for optimal fft add optimize_fft_size = true
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, optimize_fft_size = true, use_symmetry = false)


#Uncomment for the first compilation. Time consuming.
# scfres = self_consistent_field(basis, tol=1e-12, n_bands = 4, n_ep_extra = 0 );
ψ = scfres.ψ
num_bands = size(scfres.ψ[1][1,:],1);

## Centers of the gaussian guesses for silicon
Si_centers = [[-0.125,-0.125, 0.375], [0.375,-0.125,-0.125], [-0.125, 0.375,-0.125], [-0.125,-0.125,-0.125]]












########################################################
#                                                      #
##### ### #        COMPARE MATRICES          # ### #####
#                                                      #
########################################################

#For regular MP grid 4x4x4, bands at neighbours k=3 and k=62 are sperated
# So the norm of scalar products and the svd decompositions of matrices
# given by QE and Julia should be equal

# M = get_overlap(3,62,scfres.ψ,basis)

# function change_format_mmn(M,n_bands)
#     M_gf  = [a + b*im for (a,b) in [M[i,:] for i in 1:n_bands*n_bands] ]
#     M_gf = reshape(M_gf,(n_bands,n_bands))
#     M_gf
# end

# M = get_overlap(3,62,ψ,basis)
# M_gf = change_format_mmn(M,n_bands);

# #Extracted from the Mmn file generated by QE
# N = [ 0.952103093986   -0.104776535764
#     0.270260819120   -0.026743738253
#     0.019921127302   -0.006065137994
#    -0.000000003289   -0.000000001196
#     0.041747141238   -0.153803847480
#    -0.178940700709    0.631749167439
#     0.056386828231   -0.711357050949
#    -0.000000005081    0.000000001391
#     0.136598445239    0.181432616353
#    -0.402176753285   -0.546557892868
#    -0.473860663097   -0.432248777977
#     0.000000001451    0.000000007164
#    -0.000000001768   -0.000000000672
#     0.000000001590    0.000000003797
#     0.000000004355   -0.000000010637
#     0.433151277806    0.865795086489]

# N_gf = change_format_mmn(N, n_bands)


# foo,S_M,bar = svd(M_gf);
# foo,S_N,bar = svd(N_gf);
# println("Average difference between singular values : ", norm(S_M-S_N))



########################################################
#                                                      #
##### ### #         COMPARE ENERGIES         # ### #####
#                                                      #
########################################################


# Ev_to_Ha = 0.0367493
# eig = scfres.eigenvalues


# # Eigenvalues given by Quantum Espresso extracted from *.eig
# test = [    1    1   -5.523224341892
#     2    1    4.639665981716
#     3    1    5.945739453360
#     4    1    5.945739467464
#     1    2   -4.985559266513
#     2    2    3.006526873335
#     3    2    4.918387788788
#     4    2    4.999144304406
#     1    3   -3.308074842668
#     2    3   -0.516207097771
#     3    3    3.960032841790
#     4    3    4.674903178939
#     1    4   -4.477093460734
#     2    4    1.609273123690
#     3    4    3.927875522208
#     4    4    5.461823306647
#     1    5   -4.985559266408
#     2    5    3.006526871173
#     3    5    4.918387790372
#     4    5    4.999144305170
#     1    6   -4.985559265977
#     2    6    3.006526863880
#     3    6    4.918387796229
#     4    6    4.999144306307
#     1    7   -3.508801588805
#     2    7    0.390052649678
#     3    7    2.917132329383
#     4    7    4.311343637399
#     1    8   -3.508801590018
#     2    8    0.390052653671
#     3    8    2.917132330625
#     4    8    4.311343632472
#     1    9   -3.308074841241
#     2    9   -0.516207101169
#     3    9    3.960032842342
#     4    9    4.674903180364
#     1   10   -3.508801588029
#     2   10    0.390052646413
#     3   10    2.917132331249
#     4   10    4.311343638243
#     1   11   -2.435028999432
#     2   11   -0.597522301347
#     3   11    2.804967892467
#     4   11    3.555806472948
#     1   12   -2.258839222964
#     2   12   -0.699135746901
#     3   12    2.178287582772
#     4   12    3.279135815336
#     1   13   -4.477093460372
#     2   13    1.609273120826
#     3   13    3.927875523196
#     4   13    5.461823308521
#     1   14   -3.508801589558
#     2   14    0.390052651690
#     3   14    2.917132332177
#     4   14    4.311343632301
#     1   15   -2.258839224349
#     2   15   -0.699135744205
#     3   15    2.178287581321
#     4   15    3.279135815352
#     1   16   -3.935788697174
#     2   16    1.296281641106
#     3   16    3.564831711836
#     4   16    4.030455568933
#     1   17   -4.985559265977
#     2   17    3.006526863880
#     3   17    4.918387796229
#     4   17    4.999144306307
#     1   18   -4.985559266408
#     2   18    3.006526871173
#     3   18    4.918387790372
#     4   18    4.999144305170
#     1   19   -3.508801590018
#     2   19    0.390052653671
#     3   19    2.917132330625
#     4   19    4.311343632472
#     1   20   -3.508801588805
#     2   20    0.390052649678
#     3   20    2.917132329383
#     4   20    4.311343637399
#     1   21   -4.985559266513
#     2   21    3.006526873335
#     3   21    4.918387788788
#     4   21    4.999144304406
#     1   22   -5.523224341892
#     2   22    4.639665981716
#     3   22    5.945739453360
#     4   22    5.945739467464
#     1   23   -4.477093460734
#     2   23    1.609273123690
#     3   23    3.927875522208
#     4   23    5.461823306647
#     1   24   -3.308074842668
#     2   24   -0.516207097771
#     3   24    3.960032841790
#     4   24    4.674903178939
#     1   25   -3.508801589558
#     2   25    0.390052651690
#     3   25    2.917132332177
#     4   25    4.311343632301
#     1   26   -4.477093460372
#     2   26    1.609273120826
#     3   26    3.927875523196
#     4   26    5.461823308521
#     1   27   -3.935788697174
#     2   27    1.296281641106
#     3   27    3.564831711836
#     4   27    4.030455568933
#     1   28   -2.258839224349
#     2   28   -0.699135744205
#     3   28    2.178287581321
#     4   28    3.279135815352
#     1   29   -3.508801588029
#     2   29    0.390052646413
#     3   29    2.917132331249
#     4   29    4.311343638243
#     1   30   -3.308074841241
#     2   30   -0.516207101169
#     3   30    3.960032842342
#     4   30    4.674903180364
#     1   31   -2.258839222964
#     2   31   -0.699135746901
#     3   31    2.178287582772
#     4   31    3.279135815336
#     1   32   -2.435028999432
#     2   32   -0.597522301347
#     3   32    2.804967892467
#     4   32    3.555806472947
#     1   33   -3.308074835728
#     2   33   -0.516207113432
#     3   33    3.960032842780
#     4   33    4.674903185964
#     1   34   -3.508801586269
#     2   34    0.390052639385
#     3   34    2.917132336431
#     4   34    4.311343637852
#     1   35   -2.435029001478
#     2   35   -0.597522298077
#     3   35    2.804967896751
#     4   35    3.555806468168
#     1   36   -2.258839216713
#     2   36   -0.699135756925
#     3   36    2.178287578674
#     4   36    3.279135823388
#     1   37   -3.508801586584
#     2   37    0.390052640667
#     3   37    2.917132336119
#     4   37    4.311343636837
#     1   38   -4.477093458948
#     2   38    1.609273111312
#     3   38    3.927875524775
#     4   38    5.461823315926
#     1   39   -3.935788696813
#     2   39    1.296281637545
#     3   39    3.564831721856
#     4   39    4.030455563432
#     1   40   -2.258839223186
#     2   40   -0.699135744952
#     3   40    2.178287573614
#     4   40    3.279135822797
#     1   41   -2.435029002039
#     2   41   -0.597522296977
#     3   41    2.804967897224
#     4   41    3.555806466828
#     1   42   -3.935788696713
#     2   42    1.296281636627
#     3   42    3.564831724308
#     4   42    4.030455561918
#     1   43   -4.035049040064
#     2   43    0.292376632231
#     3   43    5.151346153831
#     4   43    5.151346154158
#     1   44   -2.807378909650
#     2   44   -0.400582650928
#     3   44    2.242263438305
#     4   44    4.352048771333
#     1   45   -2.258839216293
#     2   45   -0.699135757451
#     3   45    2.178287577387
#     4   45    3.279135824776
#     1   46   -2.258839221381
#     2   46   -0.699135748173
#     3   46    2.178287573778
#     4   46    3.279135824165
#     1   47   -2.807378909066
#     2   47   -0.400582652064
#     3   47    2.242263437523
#     4   47    4.352048772972
#     1   48   -2.807378907088
#     2   48   -0.400582654592
#     3   48    2.242263431449
#     4   48    4.352048779209
#     1   49   -4.477093458948
#     2   49    1.609273111312
#     3   49    3.927875524775
#     4   49    5.461823315926
#     1   50   -3.508801586584
#     2   50    0.390052640667
#     3   50    2.917132336119
#     4   50    4.311343636837
#     1   51   -2.258839223186
#     2   51   -0.699135744952
#     3   51    2.178287573614
#     4   51    3.279135822797
#     1   52   -3.935788696813
#     2   52    1.296281637545
#     3   52    3.564831721856
#     4   52    4.030455563432
#     1   53   -3.508801586269
#     2   53    0.390052639385
#     3   53    2.917132336431
#     4   53    4.311343637852
#     1   54   -3.308074835728
#     2   54   -0.516207113432
#     3   54    3.960032842780
#     4   54    4.674903185964
#     1   55   -2.258839216713
#     2   55   -0.699135756925
#     3   55    2.178287578674
#     4   55    3.279135823388
#     1   56   -2.435029001478
#     2   56   -0.597522298077
#     3   56    2.804967896751
#     4   56    3.555806468168
#     1   57   -2.258839221381
#     2   57   -0.699135748173
#     3   57    2.178287573778
#     4   57    3.279135824165
#     1   58   -2.258839216293
#     2   58   -0.699135757451
#     3   58    2.178287577387
#     4   58    3.279135824776
#     1   59   -2.807378907088
#     2   59   -0.400582654591
#     3   59    2.242263431448
#     4   59    4.352048779209
#     1   60   -2.807378909066
#     2   60   -0.400582652064
#     3   60    2.242263437523
#     4   60    4.352048772972
#     1   61   -3.935788696713
#     2   61    1.296281636627
#     3   61    3.564831724308
#     4   61    4.030455561918
#     1   62   -2.435029002039
#     2   62   -0.597522296977
#     3   62    2.804967897223
#     4   62    3.555806466829
#     1   63   -2.807378909650
#     2   63   -0.400582650928
#     3   63    2.242263438305
#     4   63    4.352048771333
#     1   64   -4.035049040064
#     2   64    0.292376632231
#     3   64    5.151346151302
#     4   64    5.151346156687][:,3]

# #Change format
# QE_eig = zeros(64,4)
# for i in 1:64
#     QE_eig[i,:] = [test[(i-1)*4 + j] for j in 1:4]
# end

# #Convert to Ha and correct shifting
# QE_eig .*= Ev_to_Ha
# shift = QE_eig[1,1] - eig[1][1]
# QE_eig .-= shift

# #test
# println.([norm(QE_eig[i,:] .-eig[i]) for i in 1:64])
