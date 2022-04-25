using ITensors
import KrylovKit 
#using KrylovKit
include("TamaTimeEvoMPS/src/TimeEvoMPS.jl")
using .TimeEvoMPS
using DelimitedFiles
#using HDF5, JLD  

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
function createMPO(sysenv, hsys::String, eps::Float64, freqfile::String, coupfile::String,)::MPO
   
   coups = readdlm(coupfile);
   freqs = readdlm(freqfile);

   NN::Int64  = size(sysenv)[1];
   NChain::Int64 = NN-1;
   
   thempo = OpSum();
   #system Hamiltonian
   #pay attention to constant S_x/y/z = 0.5 σ_x/y/z
   thempo += eps,hsys,1;
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

#Define overall system (sys+chain) and initial state``
(sysenv,psi0) = defineSystem(sys_type = "S=1/2",sys_istate = "Up",chain_size = 20, local_dim = 6);

#Create Hamiltonian specialized on sysenv MPS form
H = createMPO(sysenv,"Sz", 139.,"WSCP_MC_T0_freqs.dat","WSCP_MC_T0_coups.dat");
#show(H);

#Define quantities that must be observed
vobs = createObs([["Sx",1],["Sz",1],["N",10],["N",20] ]);

#Copy initial state intto evolving state.
psi = psi0;

#First measurement (t=0)

appo = Float64[]
for lookat in vobs
   orthogonalize!(psi,lookat.pos); 
   psidag = dag(prime(psi[lookat.pos],"Site"));
   m=scalar(psidag*op(sysenv,lookat.op,lookat.pos)*psi[lookat.pos]);
   push!(appo,m);  
end

timestep = 0.0001;
tmax = 0.0004;

cbT = LocalMeasurementCallbackTama(vobs,sysenv,timestep);
#tdvp!(psi,H,0.0001,0.4,maxdim=20,callback=cbT, progress=true)

#Disable "progress" to run on the cluster
tdvp!(psi,H,timestep,tmax,maxdim=20,callback=cbT, progress=true)

res = measurements(cbT);
times = measurement_ts(cbT);

#appo = [[i[1][1] for i in res[o]] for o in keys(res)];
poldo = [[i[1][1] for i in res[o]] for o in keys(res)];
pippo = pushfirst!(poldo,measurement_ts(cbT));

writedlm("risultati.csv",pippo);

#save("measurements.jld", "data", measurements(cbT));
#save("times.jld", "data", measurement_ts(cbT))

#???
#measurements(cbT);

#Store measurements data into JLD file. Maybe better to switch to CSV

# using PyPlot
# ts = measurement_ts(cbT)
# for o in ["Sx_1","Sz_1","N_2","N_3","N_4"]
#     S5 = getindex.(measurements(cbT)["$o"],1)
#     plot(ts,S5,"-o",label="$o")
# end
# legend()
# xlabel("t")
# norm(psi)
