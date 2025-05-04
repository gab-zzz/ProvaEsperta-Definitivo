using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using System;
using System.Threading.Tasks;

namespace Prova_Esperta;

public partial class App : Application
{
    private PythonServerManager _serverManager;

    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        base.OnFrameworkInitializationCompleted();

        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            // Aggiungi l'handler per l'evento di shutdown
            desktop.ShutdownRequested += Desktop_ShutdownRequested;
            
            // Crea e mostra la finestra principale
            var mainWindow = new MainWindow();
            desktop.MainWindow = mainWindow;
            
            // Avvia il server Python in background dopo che la UI è visualizzata
            Task.Run(async () => 
            {
                try
                {
                    // Inizializza il gestore del server Python
                    _serverManager = new PythonServerManager();
                    
                    // Avvia il server Python in modo asincrono
                    bool serverStarted = await _serverManager.StartServerAsync();
                    
                    if (!serverStarted)
                    {
                        await Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(async () => 
                        {
                            await CustomMessageBox.ShowAsync(
                                "Non è stato possibile avviare il server Python. Verifica che Python e le dipendenze siano installati correttamente.", 
                                "Errore Server");
                                
                            ShutdownApplication();
                        });
                    }
                }
                catch (Exception ex)
                {
                    await Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(async () => 
                    {
                        await CustomMessageBox.ShowAsync(
                            $"Errore durante l'avvio: {ex.Message}", 
                            "Errore");
                        ShutdownApplication();
                    });
                }
            });
        }
    }

    private void Desktop_ShutdownRequested(object sender, ShutdownRequestedEventArgs e)
    {
        // Ferma il server Python quando l'applicazione viene chiusa
        _serverManager?.Cleanup();
    }

    private void ShutdownApplication()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.Shutdown();
        }
    }
}