# train using lightning data module and lightning module 
# and save model

import pytorch_lightning as pl

from datamodule import MyDataModule
from mykobart import MyKoBartGenerator
from kobart import get_kobart_tokenizer

def configure_callbacks():
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        verbose=True
        )

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min',
        verbose=True,
        patience=3
        )
    return [early_stop, checkpoint]

if __name__ == '__main__':
    model = MyKoBartGenerator(        
        model_save_path="saved/model.pt",
        batch_size=10,
    )

    dm = MyDataModule(train_file="data/train.tsv",
                        test_file= "data/val.tsv",
                        tokenizer=get_kobart_tokenizer(),)
    
    trainer = pl.Trainer(
            gpus=1,
            distributed_backend="ddp",
            precision=16,
            amp_backend="apex",
            max_epochs=1,
            callbacks=configure_callbacks()
        )

    trainer.fit(model, dm)
