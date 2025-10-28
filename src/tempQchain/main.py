import typer

app = typer.Typer(help="TempQChain CLI")


@app.command()
def create_tb_dense(
    save_rules: bool = typer.Option(False, help="Save transitivity rules to file"),
):
    """Process TB-Dense data and create training/dev/test splits."""
    import tempQchain.data.create_tb_dense as create_tb_dense

    typer.echo("Processing TB-Dense data...")

    try:
        create_tb_dense.process_tb_dense(
            trans_rules=create_tb_dense.trans_rules,
            save_rules_to_file=save_rules,
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
    batch_size: int = typer.Option(4, help="Batch size for training"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    results_path: str = typer.Option("models/", help="Path to save models and predictions"),
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
    text_rules: bool = typer.Option(False, help="Include rules as text"),
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    optim: str = typer.Option("adamw", help="Optimizer type"),
    # Model loading/saving, experiment tracking
    loaded: bool = typer.Option(False, help="Load and evaluate existing model"),
    loaded_file: str = typer.Option("train_model", help="File name to load model from"),
    loaded_train: bool = typer.Option(False, help="Load model and continue training"),
    model_change: bool = typer.Option(False, help="Allow model architecture changes when loading"),
    save: bool = typer.Option(False, help="Save the trained model"),
    save_file: str = typer.Option("train_model", help="File name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.temporal_FR as temporal_FR

    args = argparse.Namespace(
        model=model,
        epoch=epoch,
        lr=lr,
        cuda=cuda,
        batch_size=batch_size,
        data_path=data_path,
        results_path=results_path,
        text_rules=text_rules,
        dropout=dropout,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        constraints=constraints,
        loaded=loaded,
        loaded_file=loaded_file,
        loaded_train=loaded_train,
        model_change=model_change,
        save=save,
        save_file=save_file,
        optim=optim,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
    )
    temporal_FR.main(args)


@app.command()
def temporal_yn(
    # Training parameters
    epoch: int = typer.Option(1, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    batch_size: int = typer.Option(100000, help="Batch size for training"),
    check_epoch: int = typer.Option(1, help="Check evaluation every N epochs"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    results_path: str = typer.Option("models/", help="Path to save models and predictions"),
    # Model parameters
    model: str = typer.Option("bert", help="Model type to use"),
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    use_class_weights: bool = typer.Option(False, help="Enable class weights for training"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(0.5, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(1, help="Sampling size"),
    # Additional options
    text_rules: bool = typer.Option(False, help="Include rules as text"),
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    check_condition: str = typer.Option("acc", help="Check condition: acc (accuracy) or loss"),
    # Model loading/saving and experiment tracking
    loaded: bool = typer.Option(False, help="Load and evaluate existing model"),
    loaded_file: str = typer.Option("train_model", help="File name to load model from"),
    loaded_train: bool = typer.Option(False, help="Load model and continue training"),
    save: bool = typer.Option(False, help="Save the trained model"),
    save_file: str = typer.Option("train_model", help="File name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.temporal_YN as temporal_YN

    args = argparse.Namespace(
        epoch=epoch,
        lr=lr,
        cuda=cuda,
        batch_size=batch_size,
        data_path=data_path,
        results_path=results_path,
        text_rules=text_rules,
        dropout=dropout,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        constraints=constraints,
        loaded=loaded,
        loaded_file=loaded_file,
        loaded_train=loaded_train,
        save=save,
        save_file=save_file,
        check_epoch=check_epoch,
        model=model,
        check_condition=check_condition,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
    )
    temporal_YN.main(args)


if __name__ == "__main__":
    app()
