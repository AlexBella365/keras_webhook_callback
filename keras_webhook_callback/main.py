import datetime, humanize
import requests
import keras
import telegram
import matplotlib.pyplot as plt

from io import BytesIO


class ParentCallback(keras.callbacks.Callback):

    def __init__(self,
                    model_name = 'model',
                    loss_metrics = ['loss'],
                    acc_metrics = [],
                    get_summary = False,
                    app=None):
        self.model_name = model_name
        self.loss_metrics = loss_metrics
        self.acc_metrics = acc_metrics
        self.get_summary = get_summary
        self.app = app
        self.logs_arr = []
        self.start, self.stop = None, None

    def on_train_begin(self, logs=None):
        text = '='*50+'\n'
        now = datetime.datetime.now()
        text += "Hi! your model `{:}` has started training at {:%H:%M:%S}".format(self.model_name,now)
        self.send_message(text)
        self.start = now

    def on_train_end(self, logs=None):
        now = datetime.datetime.now()
        text = "Your model `{:}` has finished training at {:%H:%M:%S}".format(self.model_name,now)
        
        self.stop = now
        delta = humanize.naturaldelta(self.stop-self.start)

        text += 'Training took {:}'.format(delta)
        self.send_message(text)

        if self.get_summary:
            summary = self.make_summary(self.logs_arr)
            self.send_message(summary)

        if self.app in ['slack']:
            return
    
        if self.loss_metrics:
            for metric in self.loss_metrics:
                plt.plot([epoch[metric] for epoch in self.logs_arr],label=f'{metric}')
                plt.legend()
        
        out = BytesIO()
        plt.savefig(fname=out,format='png')
        out.seek(0)
        self.send_message(out, type='image')
        plt.clf()

        if self.acc_metrics:
            for metric in self.acc_metrics:
                plt.plot([epoch[metric] for epoch in self.logs_arr], label=f'{metric}')
                plt.legend()
        
        out = BytesIO()
        plt.savefig(fname=out,format='png')
        out.seek(0)
        self.send_message(out, type='image')

    def on_epoch_end(self, epoch, logs=None):
        self.logs_arr.append(logs)
        if not self.get_summary:
            text = f'*Epoch* {epoch}'
            for key, value in logs.items():
                text += f'\t | \t*{key}*: {value:.3f}'
            self.send_message(text, type='text') 

    def make_summary(self, logs_arr):
        summary = ''
        for epoch, log in enumerate(logs_arr):
            summary += f'\n*Epoch*: {epoch+1}\n'
            for key, value in log.items():
                summary += f'*{key}*: {value:.3f}\n'
        return summary


class TelegramCallback(ParentCallback):

    def __init__(self,
                 bot_token = None,
                 chat_id = None,
                 model_name = 'model',
                 loss_metrics = ['loss'],
                 acc_metrics = [],
                 get_summary=False):
        ParentCallback.__init__(self,
                                model_name,
                                loss_metrics,
                                acc_metrics,
                                get_summary,
                                app='telegram')
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id

    def send_message(self, message, type='text'):
        if type == 'text':
            response = self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
        elif type == 'image':
            response = self.bot.send_photo(self.chat_id, photo=message)


class SlackCallback(ParentCallback):

    def __init__(self,
                 webhook_url = None,
                 channel = None,
                 model_name = 'model',
                 loss_metrics = ['loss'],
                 acc_metrics = [],
                 get_summary=False):
        ParentCallback.__init__(self, 
                                model_name,
                                loss_metrics, 
                                acc_metrics, 
                                get_summary,
                                app='slack')
        self.webhook_url = webhook_url
        self.channel = channel

    def send_message(self, message, type='text'):
        if type == 'text':
            payload = {
                'channel': self.channel,
                'text': message
            }
            response = requests.post(self.webhook_url, json=payload)
        elif type == 'image':
            response = self.bot.send_photo(self.chat_id, photo=message)
