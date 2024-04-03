from itertools import cycle

# Define server IPs
servers = ['192.168.1.1', '192.168.1.2', '192.168.1.3']

# Initialize a cycle iterator for round-robin load balancing
server_pool = cycle(servers)

def distribute_request():
    # Simulate incoming request from a client
    client_request = "Client Request"

    # Get the next server IP using round-robin
    next_server = next(server_pool)

    print(f"Distributing '{client_request}' to Server: {next_server}")

# Simulate incoming requests
for _ in range(10):  # Simulating 10 client requests
    distribute_request()
