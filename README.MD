# Gas Properties Calculator

Gas Properties Calculator is a Python script that calculates various properties of natural gas based on user inputs such as pressure, temperature, and gas composition.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/gas-properties-calculator.git
```

2. Navigate to the project directory:

```bash
cd gas-properties-calculator
```

## Usage

1. Open the `gasproperties.py` file in your preferred Python IDE or text editor.

2. Modify the input parameters according to your requirements:

```python
# Modify the input parameters according to your requirements
TotalGasGravity = 0.65 + 9 / 50
N2 = 2 / 100
CO2 = 8 / 100
H2S = 8 / 100
Pwh = 190
Pbh = 2600
Twh_F = 100
Twh = Twh_F + 460
Tbh_F = Twh + 70
Tbh = Tbh_F + 460
Depth = 2800
```

3. Run the script:

```bash
python gasproperties.py
```

4. View the generated plots displaying various gas properties.

## Features

- Calculates gas compressibility factor, formation volume factor, viscosity, density, and pseudo pressure.
- Generates plots illustrating the relationship between gas properties and pressure, temperature, and depth.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
