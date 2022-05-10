using ITensors
import KrylovKit 
include("../TamaTimeEvoMPS/src/TimeEvoMPS.jl")
using .TimeEvoMPS
using JSON
using DelimitedFiles

include("../TDVP_lib.jl")



#psi0 = productMPS(sysenv,stateSE);
parameters = load_pars("prova.json");

isMC = parameters["withMC"]

println(parameters["ranks_file"]);
println(parameters["times_file"]);

#Define overall system (sys+chain) and initial state``
(sysenv,psi0) = defineSystem(sys_type = "S=1/2",sys_istate = parameters["sys_ini"],
                  chain_size = parameters["chain_length"], local_dim = parameters["chain_loc_dim"]);



if(!isMC)

   #Create Hamiltonian specialized on sysenv MPS form
   println("Standard TEDOPA");
   H = createMPO(sysenv,parameters["sys_en"],parameters["sys_coup"],
               parameters["chain_freqs"],parameters["chain_coups"]);
else
   println("TEDOPA+MC");
   H = createMPO(sysenv,parameters["sys_en"],parameters["sys_coup"],
               parameters["chain_freqs"],parameters["chain_coups"],
               parameters["MC_alphas"],parameters["MC_betas"],parameters["MC_coups"],parameters["omegaInf"]);
end

#Define quantities that must be observed

loci = collect(10:10:80)
poldo = [["Sx",1],["Sz",1]]
for i in loci
   push!(poldo,["N",i+1])
end
push!(poldo,["Norm",1]);
vobs = createObs(poldo);


#Copy initial state into evolving state.
psi = copy(psi0);

#First measurement (t=0)

appo = Float64[]
for lookat in vobs
   orthogonalize!(psi,lookat.pos); 
   psidag = dag(prime(psi[lookat.pos],"Site"));
   m=scalar(psidag*op(sysenv,lookat.op,lookat.pos)*psi[lookat.pos]);
   push!(appo,m);  
end

println(appo);

timestep = parameters["tstep"];
tmax = parameters["tmax"];

cbT = LocalMeasurementCallbackTama(vobs,sysenv,parameters["ms_stride"]*timestep);

#tdvp!(psi,H,0.0001,0.4,maxdim=20,callback=cbT, progress=true)

if(!isMC)
#Disable "progress" to run on the cluster
   println("Standard TEDOPA")
   tdvp!(psi,H,timestep,tmax,maxdim=parameters["max_bond"],callback=cbT, progress=true,
   cutoff=parameters["discarded_w"],
   store_psi0 = true,io_file=parameters["out_file"],io_ranks = parameters["ranks_file"],io_times = parameters["times_file"])
else
   println("TEDOPA+MC")
   tdvpMC!(psi,H,timestep,tmax,maxdim=parameters["max_bond"],callback=cbT, progress=true,
   cutoff=parameters["discarded_w"],
   store_psi0 = true,io_file=parameters["out_file"],io_ranks = parameters["ranks_file"],io_times = parameters["times_file"])
end
