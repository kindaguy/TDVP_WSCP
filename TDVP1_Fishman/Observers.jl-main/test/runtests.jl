using Observers
using DataFrames
using Test
using JLD2

@testset "Observer" begin
  # Series for π/4
  f(k) = (-1)^(k+1)/(2k-1)
  
  function my_iterative_function(niter; observer!, observe_step)
    π_approx = 0.0
    for n in 1:niter
      π_approx += f(n)
      if iszero(n % observe_step)
        update!(observer!; π_approx = 4π_approx, iteration = n)
      end
    end
    return 4π_approx
  end
  
  # Measure the relative error from π at each iteration
  err_from_π(; π_approx, kwargs...) = abs(π - π_approx) / π
  
  # Record which iteration we are at
  iteration(; iteration, kwargs...) = iteration

  obs = Observer(["Error" => err_from_π, "Iteration" => iteration])
  
  @test length(obs) == 2
  @test results(obs, "Error") == []
  @test results(obs, "Iteration") == []
  
  niter = 10000
  observe_step = 1000
  π_approx = my_iterative_function(niter; observer! = obs, observe_step = observe_step)
  
  for res in obs
    @test length(last(last(res))) == niter ÷ observe_step
  end
  
  save("outputdata.jld2", results(obs)) 
  obs_load = load("outputdata.jld2")
  
  for (k,v) in obs_load
    @test v ≈ last(obs[k])
  end
  
  obs = Observer(["Error" => err_from_π, "Iteration" => iteration])
  obs["nofunction"] = nothing
  
  niter = 10000
  observe_step = 1000
  π_approx = my_iterative_function(niter; observer! = obs, observe_step = observe_step)
  
  @test results(obs,"nofunction") == []
  
  #
  # List syntax
  #

  obs = Observer("Error" => err_from_π, "Iteration" => iteration)
  
  @test length(obs) == 2
  @test results(obs, "Error") == []
  @test results(obs, "Iteration") == []
  
  niter = 10000
  observe_step = 1000
  π_approx = my_iterative_function(niter; observer! = obs, observe_step = observe_step)
  
  for res in obs
    @test length(last(last(res))) == niter ÷ observe_step
  end

  #
  # Function list syntax
  #

  obs = Observer(err_from_π, iteration)
  
  @test length(obs) == 2
  @test results(obs, err_from_π) == []
  @test results(obs, "err_from_π") == []
  @test results(obs, iteration) == []
  @test results(obs, "iteration") == []
  
  niter = 10000
  observe_step = 1000
  π_approx = my_iterative_function(niter; observer! = obs, observe_step = observe_step)
  
  for res in obs
    @test length(last(last(res))) == niter ÷ observe_step
  end

  f1(x::Int) = x^2
  f2(x::Int, y::Float64) = x + y
  f3(_,_,t::Tuple) = first(t)
  f4(x::Int; a::Float64) = x * a
  f5(x::Int; a::Float64, b::Float64) = x * a + b
  
  
  function my_other_iterative_function(; observer!)
    k = 2
    x = k
    y = k * √2
    t = (x+2*y,0,0)
    a = y^2
    b = 3.0
    update!(observer!, x, y, t; a = a, b = b)
  end
  
  obs = Observer(["f1" => f1, "f2" => f2, "f3" => f3, "f4" => f4, "f5" => f5])
  
  my_other_iterative_function(; observer! = obs)
  
  @test results(obs,"f1")[1] ≈ 4.0
  @test results(obs,"f2")[1] ≈ 2.0 + 2*√2
  @test results(obs,"f3")[1] ≈ 2.0 + 4*√2 
  @test results(obs,"f4")[1] ≈ 16.0
  @test results(obs,"f5")[1] ≈ 19.0
end

@testset "Observer constructed from functions" begin
  # Series for π/4
  f(k) = (-1)^(k+1)/(2k-1)
  
  function my_iterative_function(niter; observer!, observe_step)
    π_approx = 0.0
    for n in 1:niter
      π_approx += f(n)
      if iszero(n % observe_step)
        update!(observer!; π_approx = 4π_approx, iteration = n)
      end
    end
    return 4π_approx
  end
  
  # Measure the relative error from π at each iteration
  err_from_π(; π_approx, kwargs...) = abs(π - π_approx) / π
  
  # Record which iteration we are at
  iteration(; iteration, kwargs...) = iteration
  obs = Observer([err_from_π, iteration])
  
  @test length(obs) == 2
  @test results(obs, "err_from_π") == []
  @test results(obs, "iteration") == []
  
  niter = 10000
  observe_step = 1000
  π_approx = my_iterative_function(niter; observer! = obs, observe_step = observe_step)
  
  @test length(results(obs, "err_from_π")) == niter ÷ observe_step
  @test length(results(obs, "iteration")) == niter ÷ observe_step
  @test length(results(obs, err_from_π)) == niter ÷ observe_step
  @test length(results(obs, iteration)) == niter ÷ observe_step
  f1 = err_from_π
  f2 = iteration
  @test length(results(obs, f1)) == niter ÷ observe_step
  @test length(results(obs, f2)) == niter ÷ observe_step
  @test_throws KeyError results(obs, "f1")
  @test_throws KeyError results(obs, "f2")
end

@testset "save only last value" begin
  f(x::Int, y::Float64) = x + y
  g(x::Int) = x
  
  function Observers.update!(::typeof(g), results, result)
    empty!(results)
    push!(results, result)
  end
  
  function my_yet_another_iterative_function(niter::Int; observer!)
    for k in 1:niter
      x = k
      y = k * √2
      update!(observer!, x, y)
    end
  end
  
  obs = Observer(["g" => g, "f" => f])
  
  my_yet_another_iterative_function(100; observer! = obs)
  
  @test length( results(obs,"g")) == 1
  @test length( results(obs,"f")) == 100
end  

@testset "empty" begin
  f(x) = 2x
  function iterative(niter; observer!)
    for k in 1:niter
      update!(observer!, k)
    end
  end
  obs0 = Observer(["f" => f])

  obs1 = copy(obs0)
  @test obs0 == obs1
  iterative(10; observer! = obs1)
  @test obs1 ≠ obs0
  empty_results!(obs1)
  @test obs1 == obs0

  iterative(10; observer! = obs1)
  obs1 = empty_results(obs0)
  @test obs1 == obs0
end

@testset "Test element types of Array" begin
  f(k, x::Int, y::Float64) = x + y
  g(k, x) = k < 20 ? 0 : exp(im * x)

  function iterative_function(niter::Int; observer!)
    for k in 1:niter
      x = k
      y = x * √2
      update!(observer!, k, x, y)
    end
  end

  obs = Observer(["f" => f, "g" => g])
  iterative_function(100; observer! = obs)
  res = results(obs)
  @test length(res["f"]) == 100
  @test length(res["g"]) == 100
  @test res["f"] isa Vector{Float64}
  @test res["g"] isa Vector{ComplexF64}

  obs = Observer(["f" => f, "g" => g])
  iterative_function(10; observer! = obs)
  res = results(obs)
  @test length(res["f"]) == 10
  @test length(res["g"]) == 10
  @test res["f"] isa Vector{Float64}
  @test res["g"] isa Vector{Int}
end

@testset "Function Returning nothing" begin

  function sumints(niter; observer!)
    total = 0
    for n in 1:niter
      total += n
      update!(observer!; total=total, iteration=n)
    end
    return total
  end
  
  running_total(; total, kwargs...) = total

  function every_other(; total, iteration, kwargs...) 
    if iteration%2 == 0
      return total
    end
  end
  
  obs = Observer(["RunningTotal" => running_total, 
                  "EveryOther" => every_other
                 ])

  niter = 100
  total = sumints(niter; observer! = obs)

  eo = results(obs)["EveryOther"]
  rt = results(obs)["RunningTotal"]

  # Test that `nothing` does not appear in eo:
  @test findfirst(isnothing,eo) == nothing

  # Test that eo contains every other value of rt:
  @test length(eo) == div(length(rt),2)
  @test eo == rt[2:2:niter]

end

@testset "Conversion to DataFrames" begin
  f(k, x::Int, y::Float64) = x + y
  g(k, x) = k < 20 ? 0 : exp(im * x)
  function iterative_function(niter::Int; observer!)
    for k in 1:niter
      x = k
      y = x * √2
      update!(observer!, k, x, y)
    end
  end
  obs = Observer(["f" => f, "g" => g])
  iterative_function(100; observer! = obs)
  res = results(obs)
  df = DataFrame(results(obs))
  @test df.f == res["f"]
  @test df.g == res["g"]
  @test length(df.f) == 100
  @test length(df.g) == 100
  @test df.f isa Vector{Float64}
  @test df.g isa Vector{ComplexF64}
end
