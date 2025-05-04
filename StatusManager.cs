using System;

namespace Prova_Esperta
{
    public static class StatusManager
    {
        // Eventi per aggiornare l'interfaccia utente
        public static event Action<string> StatusMessageChanged;
        public static event Action<bool> ServerConnectionStatusChanged;
        public static event Action<bool> ProcessingStatusChanged;

        // Messaggi predefiniti
        public static class Messages
        {
            // Messaggi di connessione al server
            public const string ServerInitializing = "In attesa di connessione al server...";
            public const string ServerCheckingConnection = "Verifica connessione al server...";
            public const string ServerConnected = "Server connesso e pronto";
            public const string ServerTimeout = "Tempo scaduto: il server ha impiegato troppo tempo a rispondere. Verifica che il server stia funzionando correttamente e che non sia sovraccarico.";
            public const string ServerConnectionFailed = "Impossibile connettersi al server";
            public const string ServerDisconnected = "Il server è stato disconnesso";
            public const string ServerReconnectionFailed = "Errore durante la riconnessione";

            // Template di messaggi di connessione con parametri
            public const string ServerConnectionAttempt = "Tentativo {0}/{1} - Server non ancora pronto...";
            public const string ServerConnectionFailedAfterAttempts = "Impossibile connettersi al server dopo {0} tentativi";

            // Messaggi di elaborazione richieste
            public const string ProcessingRequest = "Elaborazione in corso...";
            public const string ProcessingWithRetrieval = "Risposta generata con retrieval e reasoning";
            public const string ProcessingDirect = "Risposta diretta";
            public const string ProcessingError = "Errore durante l'elaborazione";

            // Messaggi di risposta
            public const string WelcomeMessage = "Il server sta iniziando... Attendi qualche istante prima di inviare una domanda.";
            public const string ServerReadyMessage = "Server connesso. Inserisci una domanda e premi Consulta Medico per iniziare.";
            public const string ServerNotConnectedMessage = "Il server non è connesso. Impossibile elaborare la richiesta.";
            public const string ProcessingMessage = "Sto elaborando la risposta...";
            
            // Template di messaggi di errore con parametri
            public const string ErrorTemplate = "Errore: {0}";
            public const string ConnectionErrorTemplate = "Errore di connessione: {0}";
        }

        // Metodi per aggiornare i messaggi di stato
        public static void UpdateStatusMessage(string message)
        {
            StatusMessageChanged?.Invoke(message);
        }

        public static void UpdateServerConnectionStatus(bool isConnected)
        {
            ServerConnectionStatusChanged?.Invoke(isConnected);
        }

        public static void UpdateProcessingStatus(bool isProcessing)
        {
            ProcessingStatusChanged?.Invoke(isProcessing);
        }

        // Metodi di utilità per formattare messaggi con parametri
        public static string FormatConnectionAttempt(int current, int max)
        {
            return string.Format(Messages.ServerConnectionAttempt, current, max);
        }

        public static string FormatConnectionFailed(int attempts)
        {
            return string.Format(Messages.ServerConnectionFailedAfterAttempts, attempts);
        }

        public static string FormatError(string errorMessage)
        {
            return string.Format(Messages.ErrorTemplate, errorMessage);
        }

        // Gestione completa degli stati
        public static void SetServerConnecting()
        {
            UpdateStatusMessage(Messages.ServerCheckingConnection);
            UpdateServerConnectionStatus(false);
        }

        public static void SetServerConnected()
        {
            UpdateStatusMessage(Messages.ServerConnected);
            UpdateServerConnectionStatus(true);
        }

        public static void SetServerDisconnected()
        {
            UpdateStatusMessage(Messages.ServerDisconnected);
            UpdateServerConnectionStatus(false);
        }

        public static void SetError(string errorMessage)
        {
            UpdateStatusMessage(FormatError(errorMessage));
            UpdateProcessingStatus(false);
        }
    }
}