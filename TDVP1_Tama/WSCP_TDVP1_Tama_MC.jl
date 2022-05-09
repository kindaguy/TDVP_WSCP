using Pkg
Pkg.activate(".")
using ITensors
using DelimitedFiles
using TimeEvoMPS
include("./TDVP_lib.jl")

#Load parameters from JSON
parameters = load_pars("provaMC.json");

#Select TEDOPA/+MC
isMC = parameters["withMC"]

#Define overall system (sys+chain) and initial state``
(sysenv,psi0) = defineSystem(
   sys_type = "S=1/2",
   sys_istate = parameters["sys_ini"],
   chain_size = parameters["chain_length"], 
   local_dim =  parameters["chain_loc_dim"]
   );

if(!isMC)

   #Create Hamiltonian specialized on sysenv MPS form
   println("Standard TEDOPA");
   H = createMPO(
      sysenv,
      parameters["sys_en"],
      parameters["sys_coup"],
      parameters["chain_freqs"],
      parameters["chain_coups"]
      );
else
   println("TEDOPA+MC");
   H = createMPO(
      sysenv,
      parameters["sys_en"],
      parameters["sys_coup"],
      parameters["chain_freqs"],
      parameters["chain_coups"],
      parameters["MC_alphas"],
      parameters["MC_betas"],
      parameters["MC_coups"],
      parameters["omegaInf"]; 
      #This premutation in meant to enhance the action of the 
      #strongly coupled closure oscillators.
      perm = [2,1,4,3,6,5]
   );
end
                  
#Define quantities that must be observed


loci = collect(10:10:80)
poldo = [["Sx",1],["Sz",1]]
for i in loci
   push!(poldo,["N",i+1])
end
loci = collect(81:1:86)
for i in loci
    push!(poldo,["N",i+1])
end
vobs = createObs(poldo);


#Debug

#Copy initial state into evolving state.
psi,overlap = stretchBondDim(psi0,8);

#First measurement (t=0)

appo = Float64[]
for lookat in vobs
   orthogonalize!(psi,lookat.pos); 
   psidag = dag(prime(psi[lookat.pos],"Site"));
   m=scalar(psidag*op(sysenv,lookat.op,lookat.pos)*psi[lookat.pos]);
   push!(appo,m);  
end

println(appo);

flush(stdout);


timestep = parameters["tstep"];
tmax = parameters["tmax"];

cbT = LocalMeasurementCallbackTama(vobs,sysenv,parameters["ms_stride"]*timestep);

if(!isMC)
   #Disable "progress" to run on the cluster
   tdvp1!(psi,H,timestep,tmax, 
   ishermitian = true,   
   normalize = true, 
   callback=cbT, 
   progress=false, 
   exp_tol = parameters["exp_tol"],
   krylovdim = parameters["krylov_dim"],

   store_psi0 = true,

   io_file=parameters["out_file"],
   io_ranks = parameters["ranks_file"],
   io_times = parameters["times_file"]
   ) 
else
   tdvp1!(psi,H,timestep,tmax, 
   ishermitian = false, 
   issymmetric = false,  
   normalize = false, 
   callback=cbT, 
   progress=false, 
   exp_tol = parameters["exp_tol"],
   krylovdim = parameters["krylov_dim"],

   store_psi0 = true,

   io_file=parameters["out_file"],
   io_ranks = parameters["ranks_file"],
   io_times = parameters["times_file"]
   ) 

end