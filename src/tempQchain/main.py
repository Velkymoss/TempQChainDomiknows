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
def train_model(
    # Training parameters
    seed: int = typer.Option(42, help="Seed value used for experiment"),
    model: str = typer.Option("bert", help="Model used"),
    epoch: int = typer.Option(10, help="Number of training epochs"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(1e-3, help="Weight decay for AdamW"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    patience: int = typer.Option(3, help="Patience for early stopping"),
    c_lr: float = typer.Option(0.05, help="Constraint learning rate"),
    c_warmup_iters: int = typer.Option(543, help="Warm up iterations for constraint optimization"),
    c_freq_increase: int = typer.Option(10, help="Update frequency of constrained lagrange multipliers"),
    c_freq_increase_freq: int = typer.Option(1, help="Increase frequency of c_freq_increase"),
    c_lr_decay: int = typer.Option(4, help="Index for constraint learning rate decay strategy"),
    c_lr_decay_param: float = typer.Option(1.0, help="Decay parameter for constraint learning rate decay strategy"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(False, help="Enable constraints"),
    transitive_enabled: bool = typer.Option(True, help="Enable transitive constraints"),
    inverse_enabled: bool = typer.Option(True, help="Enable inverse constraints"),
    use_class_weights: bool = typer.Option(False, help="Enable class weights for training"),
    train_ratio: float = typer.Option(1.0, help="Fraction of training data to use (0.0 to 1.0)"),
    ratio_seed: int = typer.Option(42, help="Seed for sampling training data"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(1.0, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(1, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    # Model loading/saving, experiment tracking
    run_name: str = typer.Option(None, help="Run name used for MLflow and saved model"),
    best_model_dir: str = typer.Option("models/", help="Directory name to save model"),
    use_mlflow: bool = typer.Option(False, help="Use MLflow for experiment tracking"),
):
    import argparse

    import tempQchain.train as train

    args = argparse.Namespace(
        seed=seed,
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
        transitive_enabled=transitive_enabled,
        inverse_enabled=inverse_enabled,
        best_model_dir=best_model_dir,
        use_mlflow=use_mlflow,
        use_class_weights=use_class_weights,
        train_ratio=train_ratio,
        ratio_seed=ratio_seed,
        patience=patience,
        c_lr=c_lr,
        c_warmup_iters=c_warmup_iters,
        c_freq_increase=c_freq_increase,
        c_freq_increase_freq=c_freq_increase_freq,
        c_lr_decay=c_lr_decay,
        c_lr_decay_param=c_lr_decay_param,
        run_name=run_name,
    )
    train.main(args)


@app.command()
def constraint_analysis(
    # Training parameters
    seed: int = typer.Option(42, help="Seed value used for experiment"),
    model: str = typer.Option("bert", help="Model used"),
    batch_size: int = typer.Option(8, help="Batch size for analysis"),
    # Data parameters
    data_path: str = typer.Option("data/", help="Path to the data folder"),
    # Model parameters
    dropout: bool = typer.Option(False, help="Enable dropout"),
    constraints: bool = typer.Option(True, help="Enable constraints"),
    transitive_enabled: bool = typer.Option(True, help="Enable transitive constraints"),
    inverse_enabled: bool = typer.Option(True, help="Enable inverse constraints"),
    # Training method parameters
    pmd: bool = typer.Option(False, help="Use Primal Dual method"),
    beta: float = typer.Option(1.0, help="Beta parameter for PMD"),
    sampling: bool = typer.Option(False, help="Use sampling loss"),
    sampling_size: int = typer.Option(4, help="Sampling size"),
    # Additional options
    cuda: int = typer.Option(0, help="CUDA device number (-1 for CPU)"),
    output_file: str = typer.Option(
        "final_chain_questions.json", help="Path to save the extracted chain questions as a JSON array"
    ),
):
    import argparse

    import tempQchain.constraint_analysis as constraint_analysis

    args = argparse.Namespace(
        seed=seed,
        model=model,
        batch_size=batch_size,
        data_path=data_path,
        dropout=dropout,
        constraints=constraints,
        transitive_enabled=transitive_enabled,
        inverse_enabled=inverse_enabled,
        pmd=pmd,
        beta=beta,
        sampling=sampling,
        sampling_size=sampling_size,
        cuda=cuda,
    )
    constraint_analysis.main(args)


if __name__ == "__main__":
    app()
