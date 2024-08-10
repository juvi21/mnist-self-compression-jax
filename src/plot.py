import matplotlib.pyplot as plt

def plot_training_results(all_metrics, weight_count):
    plt.figure(figsize=(12, 6))
    try:
        q_values = [m['Q'] * weight_count / 8 for m in all_metrics]
        plt.plot(q_values, color="red", label="Model Size (bytes)")
    except TypeError as e:
        print(f"Error plotting Q values: {e}")
        print(f"First few Q values: {all_metrics[:5]}")
    
    try:
        accuracy_values = [m['accuracy'] * 100 for m in all_metrics]
        plt.twinx()
        plt.plot(accuracy_values, color="blue", label="Training Accuracy (%)")
    except TypeError as e:
        print(f"Error plotting accuracy values: {e}")
        print(f"First few accuracy values: {all_metrics[:5]}")

    plt.title("Model Size vs Training Accuracy Over Training")
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.show()