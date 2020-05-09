# Mechanical Hardware

The CAD for this project was done in OnShape. View the assembly and parts [here](https://cad.onshape.com/documents/a323499ca448ff5cfd3dca22/w/1d5ba33bc2795cecee47b742/e/558a06e23199a00c5437c471).

The sizes and dimensions for many of these parts will depend on the size of the TV. Take measurements of your display and make any necessary adjustments if you plan on adapting these components.

Some notes about specific parts:

* The shell is made from 0.030" Polystyrene sheets. Because the shell was designed as a sheet-metal part, the geometries can be flattened for fabrication using a laser cutter. A rigid template piece could also be created and used as a guide to trace out parts. Once the sheets are cut, they can be bent over a jig by heating the bend lines with a hot air gun. The parts were then glued together and painted.
* Because I didn't trust an interference fit to hold the large knobs on the small stepper motor D-shafts (and the exposed length of the shafts was fairly short), I machined two shaft couplers. The couplers have built in set screws to hold the motor D-shafts and fit more snuggly into the knobs.
* The knobs are 3D printed and designed to not require supports if printed face down. Consider reducing infill and increasing the wall thickness to decrease the print time.
* 3D printed brackets hold the motors to the TV. I did not want the polystyrene frame to support their weight and forces. Because TVs have no mounting features, the brackets are adhered to the TV using VHB.
* Ribs help position the shell on the TV.
* Thin spacer parts separate the Motor Boards from the motors and ensure that the magnetic sensor is correctly spaced.

## Bill of Materials

### Purchased parts

| Item | Price | Qty | Total | Link |
|---|---|---|---|---|
| TV | X | 1 | X |  |
| Raspberry Pi | $45 | 1 | $45 | [https://amazon.com/gp/product/B07JR3M7FY](https://amazon.com/gp/product/B07JR3M7FY) |
| Stepper Motors | $11 | 2 | $22 | [https://amazon.com/gp/product/B00PNEQ9T4](https://amazon.com/gp/product/B00PNEQ9T4) |
| A4988 Drivers | $9 | 1 | $9 | [https://amazon.com/gp/product/B01FFGAKK8](https://amazon.com/gp/product/B01FFGAKK8) |
| DC Power Brick | $16 | 1 | $16 | [https://amazon.com/gp/product/B01ISM267G](https://amazon.com/gp/product/B01ISM267G) |
| Polystyrene Sheets | $9 | 1 | $9 | [https://amazon.com/gp/product/B00K88Y0MI](https://amazon.com/gp/product/B00K88Y0MI) |
| microHDMI to HDMI Adapters | $8 | 1 | $8 | [https://amazon.com/gp/product/B07K21HSQX](https://amazon.com/gp/product/B07K21HSQX) |
| Jumper wires | $6 | 1 | $6 | [https://amazon.com/dp/B01L5UKAPI/](https://amazon.com/dp/B01L5UKAPI/) |
| Board Mount Screws M3 30mm Panhead | $9 | 1 | $9 | [https://amazon.com/dp/B00F33U5S6](https://amazon.com/dp/B00F33U5S6) |
| Bracket Mount Screws M3 10mm Flathead | $6 | 1 | $6 | [https://amazon.com/dp/B01D4VHJJ6](https://amazon.com/dp/B01D4VHJJ6) |
| Coupler Set Screws 4-40 1/8" Set  | $3 | 1 | $3 | [https://amazon.com/dp/B07VM9Z2FS](https://amazon.com/dp/B07VM9Z2FS) |
| Signal Cables | $12 | 2 | $24 | |

### Custom parts
| Item | Qty | Mfg. Method |
|---|---|---|---|
| Motor Board | 2 | Custom PCB |
| Breakout Board | 1 | Custom PCB |
| Knob Wheel | 2 | Print |
| Shaft Coupler | 2 | Machine |
| Bracket - Left | 1 | Print |
| Bracket - Right | 1 | Print |
| Shell | 1 | Laser cut |
| Ribs | 16+ | Laser cut |

## Assembly

1. If you will be wall-mounting the TV, complete that step first.
1. Screw the Motor Boards onto the motors with a board spacer sandwiched between each (8x M3 30mm screws).
2. Screw the motors to the brackets (8x M3 10mm screws). Keep the orientation of the board in mind; the signal and power cables will need to reach the bottom middle of the TV.
3. Attach the two shaft couplers to the motor shafts (2x 4-40 1/8" set screws)
4. Place the brackets onto the bottom of the TV using adhesive. You may need to adjust their position later.
1. Place the shell over the front of the TV so that the shaft couplers fit through the holes. Make the holes bigger if necessary, they will be covered by the knobs.
2. Slide the internal ribs into place to position the shell relative to the TV.
3. If necessary, reposition the brackets to make them symmetric.
4. Carefully press the knobs onto the shaft couplers.
5. Mount the RPi and other electronics to the back of the TV using tape or zipties. Connect the RPi to the TV's HDMI port.
6. Connect the signal and power wires from the Breakout Board to the Motor Boards

## Further work
- [ ] Add a button to toggle between AUTO and MANUAL