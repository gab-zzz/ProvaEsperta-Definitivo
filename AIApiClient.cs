using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Prova_Esperta
{
    public class AIApiClient
    {
        private readonly HttpClient client;
        private readonly string baseUrl = "http://localhost:5000";
        private const int MaxRetries = 5; // Numero massimo di tentativi per la riconnessione
        private const int RetryDelay = 5000; // Ritardo tra i tentativi in millisecondi

        public AIApiClient()
        {
            client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(20); // Timeout esteso per richieste lunghe
        }

        // Invia la domanda al server AI e restituisce la risposta e se è una query medica
        public async Task<(string risposta, bool isMedical)> SendRequestAsync(string domanda)
        {
            var requestBody = new
            {
                domanda = domanda,
                num_results = 5
            };

            var jsonRequest = JsonConvert.SerializeObject(requestBody);
            var content = new StringContent(jsonRequest, Encoding.UTF8, "application/json");

            try
            {
                HttpResponseMessage response = await client.PostAsync($"{baseUrl}/generate", content);
                response.EnsureSuccessStatusCode(); // Lancia un'eccezione se la risposta HTTP è un errore

                string responseString = await response.Content.ReadAsStringAsync();
                var responseObject = JsonConvert.DeserializeObject<dynamic>(responseString);

                if (responseObject == null)
                {
                    return ("Risposta dal server non valida o vuota.", false);
                }

                bool isMedical = false;
                string rispostaText = "";

                try
                {
                    // Estrae la risposta se presente
                    if (responseObject.risposta != null)
                    {
                        rispostaText = responseObject.risposta.ToString();
                    }
                    else
                    {
                        rispostaText = "Il server ha risposto ma il formato della risposta non è corretto.";
                    }

                    // Determina se la risposta è medica in base ai documenti utilizzati
                    if (responseObject.documenti_utilizzati != null)
                    {
                        isMedical = responseObject.documenti_utilizzati.Count > 0;
                    }
                }
                catch (Exception ex)
                {
                    // Gestione di errori nel parsing del formato JSON
                    rispostaText = $"Risposta ricevuta ma con formato inatteso: {ex.Message}\nContenuto: {responseString}";
                    isMedical = false;
                }

                return (rispostaText, isMedical);
            }
            catch (HttpRequestException ex)
            {
                // Errore HTTP nella richiesta
                string errorMessage = StatusManager.Messages.ConnectionErrorTemplate + ex.Message;
                return (errorMessage, false);
            }
            catch (TaskCanceledException)
            {
                // Timeout scaduto
                string errorMessage = StatusManager.Messages.ServerTimeout;
                return (errorMessage, false);
            }
            catch (Exception ex)
            {
                // Qualsiasi altro errore generico
                string errorMessage = ex.Message;
                return (errorMessage, false);
            }
        }

        // Verifica una singola connessione al server, utile per controlli rapidi
        public async Task TestConnectionAsync()
        {
            try
            {
                using (var tempClient = new HttpClient())
                {
                    tempClient.Timeout = TimeSpan.FromSeconds(20); // Timeout breve per il test
                    HttpResponseMessage response = await tempClient.GetAsync($"{baseUrl}/");

                    if (!response.IsSuccessStatusCode)
                    {
                        throw new Exception($"Server ha risposto con status code: {(int)response.StatusCode} {response.ReasonPhrase}");
                    }
                }
            }
            catch (TaskCanceledException)
            {
                // Timeout scaduto durante il test
                throw new Exception("Tempo scaduto durante il test di connessione al server");
            }
            catch (Exception ex)
            {
                // Fallimento del test con messaggio dettagliato
                throw new Exception(StatusManager.Messages.ServerConnectionFailed + ex.Message);
            }
        }

        // Tenta di connettersi al server con più tentativi e ritardi progressivi
        public async Task<bool> TryConnectAsync()
        {
            int attempt = 0;
            
            while (attempt < MaxRetries)
            {
                try
                {
                    await TestConnectionAsync();
                    return true; // Connessione riuscita
                }
                catch (Exception ex)
                {
                    attempt++;
                    
                    if (attempt >= MaxRetries)
                    {
                        // Dopo troppi tentativi falliti, lancia l'eccezione finale
                        string errorMessage = StatusManager.Messages.ServerConnectionFailed + ex.Message;
                        throw new Exception(errorMessage);
                    }

                    // Aspetta prima di riprovare
                    await Task.Delay(RetryDelay);
                }
            }
            
            return false; // Non dovrebbe mai essere raggiunto
        }
    }
}
