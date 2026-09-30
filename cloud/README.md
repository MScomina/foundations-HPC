## Deployment for Exercise 3

1. **Prerequisites**
   * Docker Engine 20+ and Docker Compose v2 (or the older `docker-compose` command) installed.
   * `sudo` privileges (or run Docker as a non‑root user).

2. **Setup steps**
   1. **Create the environment file**
      ```bash
      cp .env.example .env
      ```
      Open `./env` and adjust the passwords and any other sensitive values.

   2. **Start the stack**
      ```bash
      sudo docker compose up -d
      ```
      The command will pull the required images, create the volumes, and start the services.

   3. **Access Nextcloud**
      Once the containers are running, visit `http://localhost:7489` (or the host you configured). Use the credentials set in `.env`.

3. **Teardown**
   To stop and remove the containers:
   ```bash
   sudo docker compose down
   ```
   The persistent data will remain in the named volumes (`db_data` and `nextcloud_data`). Remove them with `sudo docker volume rm db_data nextcloud_data` if you want a fresh start.

---
   *All configuration values are supplied via the `.env` file; the `docker-compose.yml` contains only the service definitions.*