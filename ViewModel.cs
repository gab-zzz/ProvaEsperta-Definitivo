using System;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Prova_Esperta
{
    // ViewModel principale per l'interfaccia, usa CommunityToolkit.MVVM
    public partial class ViewModel : ObservableObject
    {
        private readonly AIApiClient _apiClient;

        [ObservableProperty] private string _domanda = string.Empty;
        [ObservableProperty] private string _risposta = string.Empty;

        private bool _isLoading;

        // Proprietà che indica se l'app sta elaborando una richiesta
        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                if (SetProperty(ref _isLoading, value))
                {
                    // Aggiorna lo stato visivo e abilita/disabilita il comando
                    StatusManager.UpdateProcessingStatus(value);
                    EseguiDomandaCommand.NotifyCanExecuteChanged();
                }
            }
        }

        private bool _isServerConnected;

        // Proprietà che indica se il server è connesso
        public bool IsServerConnected
        {
            get => _isServerConnected;
            set
            {
                if (SetProperty(ref _isServerConnected, value))
                {
                    StatusManager.UpdateServerConnectionStatus(value);
                    EseguiDomandaCommand.NotifyCanExecuteChanged();
                }
            }
        }

        [ObservableProperty] private bool _isMedicalQuery;
        [ObservableProperty] private string _statusMessage = string.Empty;

        public IAsyncRelayCommand EseguiDomandaCommand { get; }
        public IAsyncRelayCommand VerificaServerCommand { get; }

        // Costruttore del ViewModel: inizializza comandi, eventi e stato iniziale
        public ViewModel()
        {
            _apiClient = new AIApiClient();

            // Inizializza i comandi asincroni
            EseguiDomandaCommand = new AsyncRelayCommand(EseguiDomandaAsync, CanEseguiDomanda);
            VerificaServerCommand = new AsyncRelayCommand(VerificaServerConnessioneAsync);

            _isServerConnected = false;
            _isLoading = false;
            IsMedicalQuery = false;

            // Messaggi iniziali
            Risposta = StatusManager.Messages.WelcomeMessage;
            StatusMessage = StatusManager.Messages.ServerInitializing;

            // Sottoscrizione agli eventi del manager di stato
            StatusManager.StatusMessageChanged += message => StatusMessage = message;
            StatusManager.ServerConnectionStatusChanged += status => IsServerConnected = status;
            StatusManager.ProcessingStatusChanged += status => IsLoading = status;

            // Reazione al cambiamento della proprietà "Domanda"
            PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(Domanda))
                {
                    EseguiDomandaCommand.NotifyCanExecuteChanged();
                }
            };

            // Avvia subito la verifica della connessione al server
            _ = VerificaServerConnessioneAsync();
        }

        // Metodo che determina se il comando per eseguire la domanda è attivabile
        private bool CanEseguiDomanda()
        {
            return !string.IsNullOrWhiteSpace(Domanda) && !IsLoading && IsServerConnected;
        }

        // Verifica la connessione con il server API con tentativi ripetuti
        private async Task VerificaServerConnessioneAsync()
        {
            const int maxTentativi = 5;
            int tentativo = 0;

            IsLoading = true;
            StatusMessage = StatusManager.Messages.ServerCheckingConnection;

            while (tentativo < maxTentativi)
            {
                try
                {
                    // Prova a contattare il server
                    await _apiClient.TestConnectionAsync();
                    StatusMessage = StatusManager.Messages.ServerConnected;
                    Risposta = StatusManager.Messages.ServerReadyMessage;
                    IsServerConnected = true;
                    IsLoading = false;
                    return;
                }
                catch (Exception ex)
                {
                    tentativo++;
                    StatusMessage = StatusManager.FormatConnectionAttempt(tentativo, maxTentativi);

                    // Se si raggiunge il numero massimo di tentativi, segnala fallimento
                    if (tentativo == maxTentativi)
                    {
                        StatusMessage = StatusManager.FormatConnectionFailed(maxTentativi);
                        Risposta = StatusManager.FormatError(ex.Message);
                        IsServerConnected = false;
                        break;
                    }

                    // Attende 2 secondi prima di riprovare
                    await Task.Delay(2000);
                }
            }

            IsLoading = false;
        }

        // Invia la domanda al server e gestisce la risposta
        private async Task EseguiDomandaAsync()
        {
            if (!IsServerConnected)
            {
                Risposta = StatusManager.Messages.ServerNotConnectedMessage;
                return;
            }

            try
            {
                IsLoading = true;
                StatusMessage = StatusManager.Messages.ProcessingRequest;
                Risposta = StatusManager.Messages.ProcessingMessage;

                // Invia la domanda all'API e riceve la risposta
                var (risposta, isMedical) = await _apiClient.SendRequestAsync(Domanda);

                Risposta = risposta;
                IsMedicalQuery = isMedical;

                // Aggiorna lo stato in base al tipo di risposta
                StatusMessage = isMedical 
                    ? StatusManager.Messages.ProcessingWithRetrieval 
                    : StatusManager.Messages.ProcessingDirect;
            }
            catch (Exception ex)
            {
                // Gestione degli errori in caso di problemi con il server
                StatusMessage = StatusManager.Messages.ProcessingError;
                Risposta = StatusManager.FormatError(ex.Message);
                IsServerConnected = false;

                // Tenta di riconnettersi al server
                await VerificaServerConnessioneAsync();
            }
            finally
            {
                IsLoading = false;
            }
        }
    }
}
