from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# ---- Model parameters (squid giant axon, standard HH) ----
C_m  = 1.0      # membrane capacitance        (uF/cm^2)
g_Na = 120.0    # max sodium conductance      (mS/cm^2)
g_K  = 36.0     # max potassium conductance   (mS/cm^2)
g_L  = 0.3      # leak conductance            (mS/cm^2)
E_Na = 50.0     # sodium reversal potential   (mV)
E_K  = -77.0    # potassium reversal potential(mV)
E_L  = -54.4    # leak reversal potential     (mV)

# ---- Gating rate functions: alpha = opening rate, beta = closing rate, V in mV ----
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V):  return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V):  return 1.0 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80)

# ---- For any gate: steady-state target and time constant ----
def x_inf(alpha, beta): return alpha / (alpha + beta)
def tau(alpha, beta):   return 1.0 / (alpha + beta)

# ---- Injected current stimulus (uA/cm^2) as a function of time ----
def I_ext(t):
    return 10.0 if 5.0 <= t <= 30.0 else 0.0   # a step of current, 5–30 ms

# ---- The model: rate of change of [V, m, h, n] at time t ----
def derivatives(t, y):
    V, m, h, n = y

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K  = g_K  * n**4     * (V - E_K)
    I_L  = g_L             * (V - E_L)

    dVdt = (I_ext(t) - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dVdt, dmdt, dhdt, dndt]

# ---- Resting state: start each gate at its steady value for V = -65 mV ----
V0 = -65.0
m0 = x_inf(alpha_m(V0), beta_m(V0))
h0 = x_inf(alpha_h(V0), beta_h(V0))
n0 = x_inf(alpha_n(V0), beta_n(V0))
y0 = [V0, m0, h0, n0]

# Sanity check: a neuron left alone should barely move.
print(f"Resting gates:        m0={m0:.3f}  h0={h0:.3f}  n0={n0:.3f}")
print(f"Derivatives at rest:  {derivatives(0.0, y0)}")

# ---- Integrate the system over time ----
t_span = (0.0, 50.0)                  # simulate 0–50 ms
t_eval = np.linspace(*t_span, 5000)   # where to record the solution

sol = solve_ivp(derivatives, t_span, y0,
                t_eval=t_eval, method="RK45",
                rtol=1e-6, atol=1e-6)

V_trace = sol.y[0]
m_trace = sol.y[1]
h_trace = sol.y[2]
n_trace = sol.y[3]

# ---- Membrane potential with gating variables beneath ----
fig, (axV, axG) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axV.plot(sol.t, V_trace)
axV.set_ylabel("V (mV)")
axV.set_title("Membrane potential and gating variables")
axV.axhline(0, color="gray", lw=0.5, ls="--")

axG.plot(sol.t, m_trace, label="m  (Na activation)")
axG.plot(sol.t, h_trace, label="h  (Na inactivation)")
axG.plot(sol.t, n_trace, label="n  (K activation)")
axG.set_xlabel("Time (ms)")
axG.set_ylabel("Gate value")
axG.legend()

plt.tight_layout()
plt.savefig("figures/gating_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()