import typer

app = typer.Typer(help="TempQChain CLI")


@app.command()
def create_tb_dense(
    save_rules: bool = typer.Option(False, help="Save transitivity rules to file"),
    augment_train: bool = typer.Option(False, help="Augment training set with q-chains"),
):
    """Process TB-Dense data and create training/dev/test splits."""
    import tempQchain.data.create_tb_dense as create_tb_dense

    typer.echo("Processing TB-Dense data...")

    try:
        create_tb_dense.process_tb_dense(
            trans_rules=create_tb_dense.trans_rules, save_rules_to_file=save_rules, augment_train=augment_train
        )
        typer.echo("✅ Data processing completed successfully!")
    except Exception as e:
        typer.echo(f"❌ Error during data processing: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def temporal_fr(
    # Training parameters
    model: str = typer.Option("bert", help="Model used"),
    epoch: int = typer.Option(1, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(1e-3, help="Weight decay for AdamW"),
    batch_size: int = typer.Option(4, help="Batch size for training"),
    patience: int = typer.Option(3, help="Patience for early stopping"),
    c_lr: float = typer.Option(0.05, help="Constraint learning rate"),
    c_warmup_iters: int = typer.Option(150, help="Warm up iterations for constraint optimization"),
    c_freq_increase: int = typer.Option(5, help="Update frequency of constrained lagrange multipliers"),
    c_freq_increase_freq: int = typer.Option(1, help="Increase frequency of c_freq_increase"),
    c_lr_decay: int = typer.Option(0, help="Index for constraint learning rate decay strategy"),
    c_lr_decay_param: float = typer.Option(1.0, help="Decay parameter for constraint learning rate decay strategy"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    use_class_weights: bool = typer.Option(False, help="Enable class weights for training"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(0.5, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(1, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    # Model loading/saving, experiment tracking
    best_model_name: str = typer.Option("best_model", help="File name to save model"),
    best_model_dir: str = typer.Option("models/", help="File name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.temporal_fr as temporal_fr

    args = argparse.Namespace(
        model=model,
        epoch=epoch,
        lr=lr,
        weight_decay=weight_decay,
        cuda=cuda,
        batch_size=batch_size,
        data_path=data_path,
        dropout=dropout,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        constraints=constraints,
        best_model_name=best_model_name,
        best_model_dir=best_model_dir,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
        patience=patience,
        c_lr=c_lr,
        c_warmup_iters=c_warmup_iters,
        c_freq_increase=c_freq_increase,
        c_freq_increase_freq=c_freq_increase_freq,
        c_lr_decay=c_lr_decay,
        c_lr_decay_param=c_lr_decay_param,
    )
    temporal_fr.main(args)


@app.command()
def temporal_yn(
    # Training parameters
    model: str = typer.Option("bert", help="Model used"),
    epoch: int = typer.Option(1, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(1e-3, help="Weight decay for AdamW"),
    batch_size: int = typer.Option(4, help="Batch size for training"),
    patience: int = typer.Option(3, help="Patience for early stopping"),
    c_lr: float = typer.Option(0.05, help="Constraint learning rate"),
    c_warmup_iters: int = typer.Option(150, help="Warm up iterations for constraint optimization"),
    c_freq_increase: int = typer.Option(5, help="Update frequency of constrained lagrange multipliers"),
    c_freq_increase_freq: int = typer.Option(1, help="Increase frequency of c_freq_increase"),
    c_lr_decay: int = typer.Option(0, help="Index for constraint learning rate decay strategy"),
    c_lr_decay_param: float = typer.Option(1.0, help="Decay parameter for constraint learning rate decay strategy"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    use_class_weights: bool = typer.Option(False, help="Enable class weights for training"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(0.5, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(1, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    # Model loading/saving, experiment tracking
    best_model_name: str = typer.Option("best_model", help="File name to save model"),
    best_model_dir: str = typer.Option("models/", help="File name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.temporal_yn as temporal_yn

    args = argparse.Namespace(
        model=model,
        epoch=epoch,
        lr=lr,
        weight_decay=weight_decay,
        cuda=cuda,
        batch_size=batch_size,
        data_path=data_path,
        dropout=dropout,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        constraints=constraints,
        best_model_name=best_model_name,
        best_model_dir=best_model_dir,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
        patience=patience,
        c_lr=c_lr,
        c_warmup_iters=c_warmup_iters,
        c_freq_increase=c_freq_increase,
        c_freq_increase_freq=c_freq_increase_freq,
        c_lr_decay=c_lr_decay,
        c_lr_decay_param=c_lr_decay_param,
    )
    temporal_yn.main(args)


if __name__ == "__main__":
    app()
