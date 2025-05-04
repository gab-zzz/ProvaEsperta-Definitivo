using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace Prova_Esperta
{
    public class PythonServerManager
    {
        private Process _serverProcess;
        private readonly string _pythonPath;
        private readonly string _serverScript;
        private readonly int _port;
        private bool _isRunning = false;
        private readonly HttpClient _httpClient = new HttpClient();
        private const int MaxConnectionAttempts = 5;
        private const int RetryDelay = 10000; // 10 secondi di attesa tra i tentativi

        public PythonServerManager(string pythonPath = "python", string serverScript = "server.py", int port = 5000)
        {
            _pythonPath = pythonPath;
            _serverScript = serverScript;
            _port = port;
            _httpClient.Timeout = TimeSpan.FromSeconds(10);
        }

        // Avvia il server Python, se non già attivo
        public async Task<bool> StartServerAsync()
        {
            if (_isRunning)
            {
                StatusManager.SetServerConnected();
                return true;
            }

            try
            {
                StatusManager.SetServerConnecting();

                // Se il server è già attivo, non serve avviarlo
                if (await IsServerRunningAsync())
                {
                    _isRunning = true;
                    StatusManager.SetServerConnected();
                    return true;
                }

                // Altrimenti si avvia un nuovo processo
                await StartPythonServerAsync();
                return await WaitForServerToBeReadyAsync();
            }
            catch (Exception ex)
            {
                StatusManager.SetError($"Errore nell'avvio del server: {ex.Message}");
                return false;
            }
        }

        // Controlla se il server è già attivo sulla porta specificata
        private async Task<bool> IsServerRunningAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"http://localhost:{_port}");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        // Avvia il processo del server Python
        private async Task StartPythonServerAsync()
        {
            if (!File.Exists(_serverScript))
            {
                string fullPath = Path.GetFullPath(_serverScript);
                throw new FileNotFoundException($"Script server non trovato. Percorso completo: {fullPath}");
            }

            _serverProcess = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = $"-u {_serverScript}",
                    UseShellExecute = true,
                    CreateNoWindow = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false,
                    WorkingDirectory = Directory.GetCurrentDirectory()
                },
                EnableRaisingEvents = true
            };

            _serverProcess.Exited += (sender, args) =>
            {
                _isRunning = false;
                StatusManager.SetServerDisconnected();
            };

            _serverProcess.Start();
            StatusManager.SetServerConnecting();
        }

        // Attende che il server sia effettivamente pronto
        private async Task<bool> WaitForServerToBeReadyAsync()
        {
            for (int i = 0; i < MaxConnectionAttempts; i++)
            {
                try
                {
                    var response = await _httpClient.GetAsync($"http://localhost:{_port}");
                    if (response.IsSuccessStatusCode)
                    {
                        _isRunning = true;
                        StatusManager.SetServerConnected();
                        return true;
                    }
                }
                catch
                {
                    // Fallisce silenziosamente e riprova dopo il delay
                }

                await Task.Delay(RetryDelay);
            }

            throw new TimeoutException(StatusManager.Messages.ServerTimeout);
        }

        // Arresta il server se è attivo
        public void StopServer()
        {
            if (_isRunning && _serverProcess != null && !_serverProcess.HasExited)
            {
                try
                {
                    _serverProcess.Kill(true);
                    _serverProcess.Dispose();
                    _isRunning = false;
                    StatusManager.SetServerDisconnected();
                }
                catch (Exception ex)
                {
                    StatusManager.SetError($"Errore nell'arresto del server: {ex.Message}");
                }
            }
        }

        // Metodo pubblico per pulire tutto
        public void Cleanup()
        {
            StopServer();
        }
    }
}
