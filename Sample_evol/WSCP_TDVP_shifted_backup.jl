using ITensors
import KrylovKit 
#using KrylovKit
include("../TamaTimeEvoMPS/src/TimeEvoMPS.jl")
using .TimeEvoMPS
using JSON
using DelimitedFiles
#using HDF5, JLD  



function load_pars(file_name::String)
   input = open(file_name)
   s = read(input, String)
   # Aggiungo anche il nome del file alla lista di parametri.
   p =JSON.parse(s)
   return p
end

function defineSystem(;sys_type::String="S=1/2",sys_istate::String="Up",chain_size::Int64=250,local_dim::Int64=6)
   sys = siteinds(sys_type,1);
   env = siteinds("Boson",dim = local_dim, chain_size);
   sysenv = vcat(sys,env);
   
   stateSys = [sys_istate];

   #Standard approach: chain always in the vacuum state
   stateEnv = ["0" for n=1:chain_size];

   stateSE = vcat(stateSys,stateEnv);

   psi0 = productMPS(sysenv,stateSE);
   
   return (sysenv,psi0);
end


#function setInital()
function createMPO(sysenv, eps::Float64, delta::Float64, freqfile::String, coupfile::String,)::MPO
   
   coups = readdlm(coupfile);
   freqs = readdlm(freqfile);

   NN::Int64  = size(sysenv)[1];
   NChain::Int64 = NN-1;
   
   thempo = OpSum();
   #system Hamiltonian
   #pay attention to constant S_x/y/z = 0.5 σ_x/y/z
   thempo += 2*eps,"Sz",1;
   thempo += 2*delta,"Sx",1
   #system-env interaction
   #ATTENTION:
   #!Sx = 0.5 σx
   thempo += 2*coups[1],"Sx",1,"Adag",2;
   thempo += 2*coups[1],"Sx",1,"A",2;
   #chain local Hamiltonians
   for j=2:NChain
      thempo += freqs[j-1],"N",j;
   end
   
   for j=2:NChain-1
      thempo += coups[j],"A",j,"Adag",j+1;
      thempo += coups[j],"Adag",j,"A",j+1;
   end
   return MPO(thempo,sysenv);
end

function createObs(lookat)
   
   vobs = [];
   for a in lookat
      push!(vobs,opPos(a[1],a[2]));
   end

   return vobs;

end

#psi0 = productMPS(sysenv,stateSE);
parameters = load_pars("prova.json");

#Define overall system (sys+chain) and initial state``
(sysenv,psi0) = defineSystem(sys_type = "S=1/2",sys_istate = parameters["sys_ini"],
                  chain_size = parameters["chain_length"], local_dim = parameters["chain_loc_dim"]);

#Create Hamiltonian specialized on sysenv MPS form
H = createMPO(sysenv,parameters["sys_en"],parameters["sys_coup"],
               parameters["chain_freqs"],parameters["chain_coups"]);
#show(H);

#Define quantities that must be observed

loci = collect(10:10:80)
poldo = [["Sx",1],["Sz",1]]
for i in loci
   push!(poldo,["N",i])
end
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

timestep = parameters["tstep"];
tmax = parameters["tmax"];

cbT = LocalMeasurementCallbackTama(vobs,sysenv,parameters["ms_stride"]*timestep);
#tdvp!(psi,H,0.0001,0.4,maxdim=20,callback=cbT, progress=true)

#Disable "progress" to run on the cluster
tdvp!(psi,H,timestep,tmax,maxdim=parameters["max_bond"],callback=cbT, progress=false,cutoff=parameters["discarded_w"],io_file=parameters["out_file"],store_psi0 = true)

