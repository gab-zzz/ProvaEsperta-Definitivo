using Avalonia;
using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using System.Threading.Tasks;

namespace Prova_Esperta
{
    // Finestra personalizzata per mostrare messaggi all'utente con diversi pulsanti e risultati
    public class CustomMessageBox : Window
    {
        // Enum per indicare quali pulsanti visualizzare
        public enum MessageBoxButtons
        {
            OK,
            OKCancel,
            YesNo,
            YesNoCancel
        }

        // Enum per indicare quale pulsante è stato premuto
        public enum MessageBoxResult
        {
            OK,
            Cancel,
            Yes,
            No
        }

        private TaskCompletionSource<MessageBoxResult> _resultSource;

        public CustomMessageBox()
        {
            // Impostazioni di base della finestra
            this.Width = 400;
            this.Height = 200;
            this.CanResize = false;
            this.WindowStartupLocation = WindowStartupLocation.CenterScreen;
        }

        // Metodo statico per mostrare la finestra e ottenere un risultato asincrono
        public static Task<MessageBoxResult> ShowAsync(
            string message, 
            string title = "Messaggio", 
            MessageBoxButtons buttons = MessageBoxButtons.OK)
        {
            // Crea la finestra con titolo specificato
            var messageBox = new CustomMessageBox
            {
                Title = title
            };

            // Stack principale che contiene il messaggio e i pulsanti
            var mainPanel = new StackPanel
            {
                Margin = new Thickness(20),
                Spacing = 20
            };

            // Blocco di testo con il messaggio da mostrare
            var messageTextBlock = new TextBlock
            {
                Text = message,
                TextWrapping = TextWrapping.Wrap,
                VerticalAlignment = VerticalAlignment.Center
            };
            mainPanel.Children.Add(messageTextBlock);

            // Stack orizzontale per i pulsanti
            var buttonPanel = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Spacing = 10
            };

            // Metodo locale per aggiungere pulsanti e gestire il click
            void AddButton(string text, MessageBoxResult result)
            {
                var button = new Button
                {
                    Content = text,
                    Width = 80
                };
                button.Click += (s, e) =>
                {
                    messageBox._resultSource.SetResult(result); // Imposta il risultato
                    messageBox.Close();                         // Chiude la finestra
                };
                buttonPanel.Children.Add(button);
            }

            // Aggiunge i pulsanti in base all'opzione scelta
            switch (buttons)
            {
                case MessageBoxButtons.OK:
                    AddButton("OK", MessageBoxResult.OK);
                    break;
                case MessageBoxButtons.OKCancel:
                    AddButton("OK", MessageBoxResult.OK);
                    AddButton("Annulla", MessageBoxResult.Cancel);
                    break;
                case MessageBoxButtons.YesNo:
                    AddButton("Sì", MessageBoxResult.Yes);
                    AddButton("No", MessageBoxResult.No);
                    break;
                case MessageBoxButtons.YesNoCancel:
                    AddButton("Sì", MessageBoxResult.Yes);
                    AddButton("No", MessageBoxResult.No);
                    AddButton("Annulla", MessageBoxResult.Cancel);
                    break;
            }

            mainPanel.Children.Add(buttonPanel);
            messageBox.Content = mainPanel;

            // Inizializza l'oggetto per attendere il risultato asincrono
            messageBox._resultSource = new TaskCompletionSource<MessageBoxResult>();

            // Gestione della chiusura finestra: se l'utente chiude la finestra manualmente
            messageBox.Closing += (s, e) =>
            {
                if (!messageBox._resultSource.Task.IsCompleted)
                    messageBox._resultSource.SetResult(MessageBoxResult.Cancel);
            };

            // Mostra la finestra modale (non associata a una finestra padre)
            _ = messageBox.ShowDialog(null);

            return messageBox._resultSource.Task;
        }
    }
}
