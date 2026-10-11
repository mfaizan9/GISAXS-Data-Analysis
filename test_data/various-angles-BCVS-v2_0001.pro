<?xml version="1.0" encoding="UTF-8"?> <BornAgain Version = "1.7.1" ><InstrumentModel Name = "DefaultName" >
        <Item ModelType = "Instrument" Tag = "" DisplayName = "Instrument" >
            <Item ModelType = "Property" Tag = "Name" DisplayName = "Name" >
                <Parameter ParType = "QString" ParRole = "0" ParValue = "AntonPaar" />
            </Item >
            <Item ModelType = "Property" Tag = "Identifier" DisplayName = "Identifier" >
                <Parameter ParType = "QString" ParRole = "0" ParValue = "91C7EACC-550E-4525-9AF7-0DB9D5320C5D" />
            </Item ><Item ModelType = "Detector" Tag = "" DisplayName = "Detector" >
                <Item ModelType = "GroupProperty" Tag = "DetectorType" DisplayName = "DetectorType" >
                    <Parameter ParType = "GroupProperty_t" ParRole = "0" ParValue = "RectangularDetector" />
                    <Item ModelType = "RectangularDetector" Tag = "" DisplayName = "RectangularDetector" >
                        <Item ModelType = "BasicAxis" Tag = "X axis" DisplayName = "X axis" >
                            <Item ModelType = "Property" Tag = "Visibility" DisplayName = "Visibility" >
                                <Parameter ParType = "bool" ParRole = "0" ParValue = "1" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Nbins" DisplayName = "Nbins" >
                                <Parameter ParType = "int" ParRole = "0" ParValue = "1028" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Min" DisplayName = "Min" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "0.000000000000e+0" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Max" DisplayName = "Width" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "77.025" />
                            </Item >
                            <Item ModelType = "Property" Tag = "title" DisplayName = "title" >
                                <Parameter ParType = "QString" ParRole = "0" ParValue = "" />
                            </Item >
                        </Item >
                        <Item ModelType = "BasicAxis" Tag = "Y axis" DisplayName = "Y axis" >
                            <Item ModelType = "Property" Tag = "Visibility" DisplayName = "Visibility" >
                                <Parameter ParType = "bool" ParRole = "0" ParValue = "1" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Nbins" DisplayName = "Nbins" >
                                <Parameter ParType = "int" ParRole = "0" ParValue = "1062" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Min" DisplayName = "Min" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "0.000000000000e+0" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Max" DisplayName = "Height" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "79.575" />
                            </Item >
                            <Item ModelType = "Property" Tag = "title" DisplayName = "title" >
                                <Parameter ParType = "QString" ParRole = "0" ParValue = "" />
                            </Item >
                        </Item >
                        <Item ModelType = "Property" Tag = "Alignment" DisplayName = "Alignment" >
                            <Parameter ParType = "ComboProperty" ParRole = "0" ParValue = "Generic" />
                        </Item >
                        <Item ModelType = "Vector" Tag = "Normal vector" DisplayName = "Normal vector" >
                            <Parameter ParType = "QString" ParRole = "0" ParValue = "(0, 0, 0)" />
                            <Item ModelType = "Property" Tag = "X" DisplayName = "X" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "1599.95648541456" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Y" DisplayName = "Y" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "0" />
                            </Item >
                            <Item ModelType = "Property" Tag = "Z" DisplayName = "Z" >
                                <Parameter ParType = "double" ParRole = "0" ParValue = "-11.4976182285881" />
                            </Item >
                        </Item >
                        <Item ModelType = "Property" Tag = "u0" DisplayName = "u0" >
                            <Parameter ParType = "double" ParRole = "0" ParValue = "38.2304231601995" />
                        </Item >
                        <Item ModelType = "Property" Tag = "v0" DisplayName = "v0" >
                            <Parameter ParType = "double" ParRole = "0" ParValue = "-1.53121965012417" />
                        </Item >
                    </Item >
                </Item >
            </Item ><Item ModelType="Beam" Tag="" DisplayName="Beam">
               <Item ModelType = "Property" Tag = "Intensity" DisplayName = "Intensity" >
                   <Parameter ParType = "ScientificDoubleProperty" ParRole = "0" ParValue = "1e+8" />
                </Item >
               <Item ModelType = "BeamWavelength" Tag = "Wavelength" DisplayName = "Wavelength" >
                   <Item ModelType = "GroupProperty" Tag = "Distribution" DisplayName = "Distribution" >
                       <Parameter ParType = "GroupProperty_t" ParRole = "0" ParValue = "DistributionNone" />
                       <Item ModelType = "DistributionNone" Tag = "" DisplayName = "DistributionNone" >
                           <Item ModelType = "Property" Tag = "is initialized" DisplayName = "is initialized" >
                               <Parameter ParType = "bool" ParRole = "0" ParValue = "0" />
                            </Item >
                           <Item ModelType = "Property" Tag = "Value" DisplayName = "Value" >
                               <Parameter ParType = "double" ParRole = "0" ParValue = " 0.15418" />
                            </Item >
                        </Item >
                    </Item >
                </Item >
               <Item ModelType = "BeamInclinationAngle" Tag = "InclinationAngle" DisplayName = "InclinationAngle" >
                   <Item ModelType = "GroupProperty" Tag = "Distribution" DisplayName = "Distribution" >
                       <Parameter ParType = "GroupProperty_t" ParRole = "0" ParValue = "DistributionNone" />
                       <Item ModelType = "DistributionNone" Tag = "" DisplayName = "DistributionNone" >
                           <Item ModelType = "Property" Tag = "Value" DisplayName = "Value" >
                               <Parameter ParType = "double" ParRole = "0" ParValue = " 0.299998955308891" />
                            </Item >
                        </Item >
                    </Item >
                </Item >
            </Item > </Item> </InstrumentModel > <RealDataModel Name="DefaultName">
        <Item ModelType = "RealData" Tag = "" DisplayName = "RealData" >
            <Item ModelType = "Property" Tag = "Name" DisplayName = "Name" >
                <Parameter ParType = "QString" ParRole = "0" ParValue = "various-angles-BCVS-v2_0001.txt.gz" />
            </Item >
            <Item ModelType = "Property" Tag = "Instrument Id" DisplayName = "Instrument Id" >
                <Parameter ParType = "QString" ParRole = "0" ParValue = "91C7EACC-550E-4525-9AF7-0DB9D5320C5D" />
            </Item >
            <Item ModelType = "IntensityData" Tag = "" DisplayName = "IntensityData" >
                <Item ModelType = "Property" Tag = "FileName" DisplayName = "FileName" >
                    <Parameter ParType = "QString" ParRole = "0" ParValue = "various-angles-BCVS-v2_0001.txt.gz" />
                </Item >
            </Item >
        </Item >
    </RealDataModel ></BornAgain>