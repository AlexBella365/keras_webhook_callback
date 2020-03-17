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
        self.times = {}

    def send_message(self,*args, **kwargs):
        '''
        To be overwritten by the children classes
        '''
        pass
        
    def on_train_begin(self, logs=None):
        text = '*'*30+'\n'
        now = datetime.datetime.now()
        text += "Keras model `{:}` has started training at {:%H:%M:%S}.".format(self.model_name,now)
        self.send_message(text)
        self.times['training_start'] = now

    def on_train_end(self, logs=None):
        now = datetime.datetime.now()
        text = "Your model `{:}` has finished training at {:%H:%M:%S}.".format(self.model_name,now)
        
        self.times['training_stop'] = now
        delta = humanize.naturaldelta(self.times['training_stop']-self.times['training_start'])

        text += ' Training took {:}.'.format(delta)
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

    def on_epoch_begin(self, epoch, logs=None):
        self.times['epoch_start'] = datetime.datetime.now()
    
    def on_epoch_end(self, epoch, logs=None):
        self.logs_arr.append(logs)
        self.times['epoch_stop'] = datetime.datetime.now()

        if not self.get_summary:
            delta = humanize.naturaldelta(self.times['epoch_stop'] - self.times['epoch_start'])
            text = f'*Epoch* {epoch:0>3} (time elapsed: {delta})\n'
            for key, value in logs.items():
                text += f'\t| *{key:10}*: {value:.3f}\n'
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
            self.bot.send_message(self.chat_id, message, parse_mode='Markdown')
        elif type == 'image':
            self.bot.send_photo(self.chat_id, photo=message)


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
            requests.post(self.webhook_url, json=payload)
        elif type == 'image':
            self.bot.send_photo(self.chat_id, photo=message)
