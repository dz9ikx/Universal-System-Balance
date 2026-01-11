import numpy as np
import matplotlib.pyplot as plt

def detect_crisis_points(data_series, zeta_c=1.0 / np.sqrt(2)):
    """
    Detects critical points (t_crit) where the system is likely to undergo 
    a phase transition, based on the 0.707 invariant theory.

    A crisis point is detected when:
    1) The third derivative (jerk) is zero (d3x/dt3 = 0).
    2) The normalized acceleration magnitude exceeds the zeta_c threshold.
    """
    # 1. Calculate derivatives: velocity, acceleration (a), and jerk (j)
    # Using numpy.diff for simplicity, note that this shortens the array by 1 each time.
    velocity = np.diff(data_series, n=1, axis=0)
    acceleration = np.diff(velocity, n=1, axis=0)
    jerk = np.diff(acceleration, n=1, axis=0)
    
    # Align time indices for plotting (original data is longer)
    time_aligned = data_series[3:] 

    # 2. Find points where jerk changes sign (d3x/dt3 approx 0)
    # We look for sign changes between consecutive points
    jerk_sign_changes = np.diff(np.sign(jerk), n=1, axis=0) != 0
    
    # Filter acceleration data to match the length of sign_changes
    accel_aligned = acceleration[1:]
    
    # 3. Normalize acceleration
    rms_a = np.sqrt(np.mean(accel_aligned**2))
    normalized_a = np.abs(accel_aligned) / rms_a

    # 4. Identify crisis points based on both criteria
    # We combine the boolean mask for sign changes with the threshold condition
    is_crisis = (jerk_sign_changes.flatten()) & (normalized_a > zeta_c).flatten()
    
    # Return the indices/times where a crisis is detected
    return np.where(is_crisis)[0] + 3 # Adjust indices back to original time frame

# Example usage (simulating a complex system signal):
if __name__ == '__main__':
    # Simulate a signal with some noise and a sudden shift (crisis)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(t * 2 * np.pi * 0.5) + np.random.normal(0, 0.5, 1000)
    # Add a crisis event around t=7
    signal[700:] += np.linspace(0, 5, 300)**2

    crisis_indices = detect_crisis_points(signal)
    
    print(f"Detected {len(crisis_indices)} crisis points.")
    # print(f"Crisis indices: {crisis_indices}")

    # Visualization of the results
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='System Signal (Simulated)')
    plt.scatter(t[crisis_indices], signal[crisis_indices], color='red', marker='x', s=100, zorder=10, label='Detected Crisis Points ($t_{crit}$)')
    plt.axhline(y=np.mean(signal), color='gray', linestyle='--', linewidth=1, label='Mean Signal Level')
    plt.title('Crisis Detection using the Universal System Balance Theory')
    plt.xlabel('Time')
    plt.ylabel('Amplitude / State')
    plt.legend()
    plt.grid(True)
    plt.show()

