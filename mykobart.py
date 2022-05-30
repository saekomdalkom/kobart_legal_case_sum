from lightningbase import LightningBase
from transformers import BartForConditionalGeneration
from kobart import get_pytorch_kobart_model
import torch

class MyKoBartGenerator(LightningBase):
    def __init__(
        self,
        model_save_path: str,
        batch_size: int,
        max_len: int = 512,
        # num_gpus: int,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        save_step_interval: int = 1000,
        # accelerator: str = "ddp",
        # precision: int = 16,
        # use_amp: bool = True,
    ) -> None:
        super(MyKoBartGenerator, self).__init__(
            model_save_path=model_save_path,
            max_len=max_len,
            batch_size=batch_size,
            # num_gpus=num_gpus,
            lr=lr,
            weight_decay=weight_decay,
            save_step_interval=save_step_interval,
            # accelerator=accelerator,
            # precision=precision,
            # use_amp=use_amp,
        )

        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.train()
        self.pad_token_id = 0

    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('train_loss', loss, prog_bar=True)
        self.save_model()
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.save_model()
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)