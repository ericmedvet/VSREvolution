package it.units.erallab.locomotion;

import it.units.erallab.builder.FunctionGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.SpikingQuantizedFunctionGrid;
import it.units.erallab.builder.phenotype.*;
import it.units.erallab.builder.phenotype.learningSNN.*;
import it.units.erallab.builder.robot.FixedHeteroQuantizedSpikingDistributed;
import it.units.erallab.builder.robot.FixedHeteroSpikingDistributed;
import it.units.erallab.builder.robot.FixedHomoQuantizedSpikingDistributed;
import it.units.erallab.builder.robot.FixedHomoSpikingDistributed;
import it.units.erallab.hmsrobots.core.controllers.snn.IzhikevicNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuronWithHomeostasis;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.AverageFrequencySpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.MovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.UniformWithMemoryValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedIzhikevicNeuron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLIFNeuron;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedLIFNeuronWithHomeostasis;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.QuantizedSpikingFunction;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedAverageFrequencySpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedMovingAverageSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.stv.QuantizedSpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedUniformValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedUniformWithMemoryValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snndiscr.converters.vts.QuantizedValueToSpikeTrainConverter;

import java.util.Map;
import java.util.function.BiFunction;

import static it.units.erallab.hmsrobots.util.Utils.params;

public class SnnUtils {

  @SuppressWarnings({"unchecked", "rawtypes"})
  public static PrototypedFunctionBuilder<?, ?> getMapperBuilderFromName(String name) {
    String fixedHomoSpikingDistributed = "fixedHomoSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedHeteroSpikingDistributed = "fixedHeteroSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedHomoQuantizedSpikingDistributed = "fixedHomoQuantSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String fixedHeteroQuantizedSpikingDistributed = "fixedHeteroQuantSpikeDist-(?<nSignals>\\d+)" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String msn = "MSNd-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?";
    String quantizedMsn = "QMSNd-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?";
    String msnWithConverter = "MSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h|lif_h_output|lif_h_io))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedMsnWithConverter = "QMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h|lif_h_output|lif_h_io))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String oneHotMsnWithConverter = "HMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h|lif_h_output|lif_h_io))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?";
    String learningMsnWithConverter = "LMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedNumericLearningMsnWithConverter = "QNLMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters = "QNLFVMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)" +
        "-(?<symmParam1>\\d+(\\.\\d+))-(?<symmParam2>\\d+(\\.\\d+))-(?<symmParam3>\\d+(\\.\\d+))-(?<symmParam4>\\d+(\\.\\d+))" +
        "-(?<asymmParam1>\\d+(\\.\\d+))-(?<asymmParam2>\\d+(\\.\\d+))-(?<asymmParam3>\\d+(\\.\\d+))-(?<asymmParam4>\\d+(\\.\\d+))";
    String quantizedNumericLearningFixedPoolMsnWithConverter = "QNLFMSN-(?<nRules>\\d+)-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedHebbianNumericLearningMsnWithConverter = "QHLMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedHebbianNumericLearningWeightsMSNWithConverters = "QHLWMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedBinaryAndDoublesLearningMsnWithConverterAndTuning = "QBDLTMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";
    String quantizedBinaryAndDoublesLearningMsnWithConverter = "QBDLMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)" +
        "-(?<symmParam1>\\d+(\\.\\d+))-(?<symmParam2>\\d+(\\.\\d+))-(?<symmParam3>\\d+(\\.\\d+))-(?<symmParam4>\\d+(\\.\\d+))" +
        "-(?<asymmParam1>\\d+(\\.\\d+))-(?<asymmParam2>\\d+(\\.\\d+))-(?<asymmParam3>\\d+(\\.\\d+))-(?<asymmParam4>\\d+(\\.\\d+))";
    String oneHotBinaryAndDoublesLearningMsnWithConverter = "LHMSN-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<symmParam1>\\d+(\\.\\d+))-(?<symmParam2>\\d+(\\.\\d+))-(?<symmParam3>\\d+(\\.\\d+))-(?<symmParam4>\\d+(\\.\\d+))" +
        "-(?<asymmParam1>\\d+(\\.\\d+))-(?<asymmParam2>\\d+(\\.\\d+))-(?<asymmParam3>\\d+(\\.\\d+))-(?<asymmParam4>\\d+(\\.\\d+))";
    String quantizedHebbianNumericLearningClippedWeightsMSNWithConverters = "QHLCWMSN-(?<maxWeight>\\d+(\\.\\d+)?)-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<spikeType>(lif|iz|lif_h))" +
        "(-(?<lRestPot>-?\\d+(\\.\\d+)?)-(?<lThreshPot>-?\\d+(\\.\\d+)?)-(?<lambda>\\d+(\\.\\d+)?)(-(?<theta>\\d+(\\.\\d+)?))?)?" +
        "(-(?<izParams>(regular_spiking_params)))?" +
        "-(?<iConv>(unif|unif_mem))-(?<iFreq>\\d+(\\.\\d+)?)-(?<oConv>(avg|avg_mem))(-(?<oMem>\\d+))?-(?<oFreq>\\d+(\\.\\d+)?)";

    String spikingFunctionGrid = "snnFuncGrid-(?<innerMapper>.*)";
    String spikingQuantizedFunctionGrid = "snnQuantFuncGrid-(?<innerMapper>.*)";

    Map<String, String> params;
    //robot mappers
    if ((params = params(fixedHomoSpikingDistributed, name)) != null) {
      ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
      SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHomoSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    if ((params = params(fixedHeteroSpikingDistributed, name)) != null) {
      ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
      SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHeteroSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    if ((params = params(fixedHomoQuantizedSpikingDistributed, name)) != null) {
      QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
      QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHomoQuantizedSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    if ((params = params(fixedHeteroQuantizedSpikingDistributed, name)) != null) {
      QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
      QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
      if (params.containsKey("iConv")) {
        switch (params.get("iConv")) {
          case "unif":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
            }
            break;
          case "unif_mem":
            if (params.containsKey("iFreq")) {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
            } else {
              valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter();
            }
            break;
        }
      }
      if (params.containsKey("oConv")) {
        switch (params.get("oConv")) {
          case "avg":
            if (params.containsKey("oFreq")) {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
            } else {
              spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
            }
            break;
          case "avg_mem":
            if (params.containsKey("oFreq")) {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                    Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              }
            } else {
              if (params.containsKey("oMem")) {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
              } else {
                spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter();
              }
            }
            break;
        }
      }
      return new FixedHeteroQuantizedSpikingDistributed(
          Integer.parseInt(params.get("nSignals")),
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    }
    //function mappers
    if ((params = params(msn, name)) != null || (params = params(msnWithConverter, name)) != null || (params = params(learningMsnWithConverter, name)) != null) {
      BiFunction<Integer, Integer, SpikingFunction> neuronBuilder = null;
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          neuronBuilder = (l, n) -> new LIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> new LIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> new LIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta);
        } else {
          neuronBuilder = (l, n) -> new LIFNeuronWithHomeostasis();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_output")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new LIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new LIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new LIFNeuronWithHomeostasis() : new LIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_io")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new LIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new LIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new LIFNeuronWithHomeostasis() : new LIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("iz")) {
        if (params.containsKey("izParams")) {
          IzhikevicNeuron.IzhikevicParameters izhikevicParameters = IzhikevicNeuron.IzhikevicParameters.valueOf(params.get("izParams").toUpperCase());
          neuronBuilder = (l, n) -> new IzhikevicNeuron(izhikevicParameters);
        } else {
          neuronBuilder = (l, n) -> new IzhikevicNeuron();
        }
      }
      if ((params = params(msn, name)) != null) {
        return new MSN(
            Double.parseDouble(params.get("ratio")),
            Integer.parseInt(params.get("nLayers")),
            neuronBuilder
        );
      }
      if ((params = params(msnWithConverter, name)) != null || (params = params(learningMsnWithConverter, name)) != null) {
        ValueToSpikeTrainConverter valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
        SpikeTrainToValueConverter spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
        if (params.containsKey("iConv")) {
          switch (params.get("iConv")) {
            case "unif":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new UniformValueToSpikeTrainConverter();
              }
              break;
            case "unif_mem":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new UniformWithMemoryValueToSpikeTrainConverter();
              }
              break;
          }
        }
        if (params.containsKey("oConv")) {
          switch (params.get("oConv")) {
            case "avg":
              if (params.containsKey("oFreq")) {
                spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              } else {
                spikeTrainToValueConverter = new AverageFrequencySpikeTrainToValueConverter();
              }
              break;
            case "avg_mem":
              if (params.containsKey("oFreq")) {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                      Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
                }
              } else {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new MovingAverageSpikeTrainToValueConverter();
                }
              }
              break;
          }
        }
        if ((params = params(msnWithConverter, name)) != null)
          return new MSNWithConverter(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        if ((params = params(learningMsnWithConverter, name)) != null) {
          return new LearningMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
      }
    }

    if ((params = params(quantizedMsn, name)) != null ||
        (params = params(oneHotMsnWithConverter, name)) != null ||
        (params = params(quantizedMsnWithConverter, name)) != null ||
        (params = params(quantizedNumericLearningMsnWithConverter, name)) != null ||
        (params = params(quantizedNumericLearningFixedPoolMsnWithConverter, name)) != null ||
        (params = params(quantizedHebbianNumericLearningMsnWithConverter, name)) != null ||
        (params = params(quantizedHebbianNumericLearningWeightsMSNWithConverters, name)) != null ||
        (params = params(quantizedBinaryAndDoublesLearningMsnWithConverterAndTuning, name)) != null ||
        (params = params(quantizedHebbianNumericLearningClippedWeightsMSNWithConverters, name)) != null ||
        (params = params(quantizedBinaryAndDoublesLearningMsnWithConverter, name)) != null ||
        (params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null ||
        (params = params(oneHotBinaryAndDoublesLearningMsnWithConverter, name)) != null
    ) {
      BiFunction<Integer, Integer, QuantizedSpikingFunction> neuronBuilder = null;
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          neuronBuilder = (l, n) -> new QuantizedLIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> new QuantizedLIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h")) {
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> new QuantizedLIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta);
        } else {
          neuronBuilder = (l, n) -> new QuantizedLIFNeuronWithHomeostasis();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_output")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new QuantizedLIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new QuantizedLIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex) ? new QuantizedLIFNeuronWithHomeostasis() : new QuantizedLIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("lif_h_io")) {
        int outputLayerIndex = Integer.parseInt(params.get("nLayers")) + 1;
        if (params.containsKey("lRestPot") && params.containsKey("lThreshPot") && params.containsKey("lambda") && params.containsKey("theta")) {
          double restingPotential = Double.parseDouble(params.get("lRestPot"));
          double thresholdPotential = Double.parseDouble(params.get("lThreshPot"));
          double lambda = Double.parseDouble(params.get("lambda"));
          double theta = Double.parseDouble(params.get("theta"));
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new QuantizedLIFNeuronWithHomeostasis(restingPotential, thresholdPotential, lambda, theta) : new QuantizedLIFNeuron(restingPotential, thresholdPotential, lambda);
        } else {
          neuronBuilder = (l, n) -> (l == outputLayerIndex || l == 0) ? new QuantizedLIFNeuronWithHomeostasis() : new QuantizedLIFNeuron();
        }
      }
      if (params.containsKey("spikeType") && params.get("spikeType").equals("iz")) {
        if (params.containsKey("izParams")) {
          QuantizedIzhikevicNeuron.IzhikevicParameters izhikevicParameters = QuantizedIzhikevicNeuron.IzhikevicParameters.valueOf(params.get("izParams").toUpperCase());
          neuronBuilder = (l, n) -> new QuantizedIzhikevicNeuron(izhikevicParameters);
        } else {
          neuronBuilder = (l, n) -> new QuantizedIzhikevicNeuron();
        }
      }
      if ((params = params(quantizedMsn, name)) != null) {
        return new QuantizedMSN(
            Double.parseDouble(params.get("ratio")),
            Integer.parseInt(params.get("nLayers")),
            neuronBuilder
        );
      }
      if ((params = params(oneHotMsnWithConverter, name)) != null) {
        return new OneHotMSNWithConverters(
            Double.parseDouble(params.get("ratio")),
            Integer.parseInt(params.get("nLayers")),
            neuronBuilder,
            5, 5
        );
      }
      if ((params = params(oneHotBinaryAndDoublesLearningMsnWithConverter, name)) != null) {
        double[] symmetricParams = {Double.parseDouble(params.get("symmParam1")), Double.parseDouble(params.get("symmParam2")),
            Double.parseDouble(params.get("symmParam3")), Double.parseDouble(params.get("symmParam4"))};
        double[] asymmetricParams = {Double.parseDouble(params.get("asymmParam1")), Double.parseDouble(params.get("asymmParam2")),
            Double.parseDouble(params.get("asymmParam3")), Double.parseDouble(params.get("asymmParam4"))};
        return new LearningOneHotMSNWithConverters(
            Double.parseDouble(params.get("ratio")),
            Integer.parseInt(params.get("nLayers")),
            neuronBuilder,
            5, 5,
            symmetricParams, asymmetricParams
        );
      }
      if ((params = params(quantizedMsnWithConverter, name)) != null ||
          (params = params(quantizedNumericLearningMsnWithConverter, name)) != null ||
          (params = params(quantizedNumericLearningFixedPoolMsnWithConverter, name)) != null ||
          (params = params(quantizedHebbianNumericLearningMsnWithConverter, name)) != null ||
          (params = params(quantizedHebbianNumericLearningWeightsMSNWithConverters, name)) != null ||
          (params = params(quantizedBinaryAndDoublesLearningMsnWithConverterAndTuning, name)) != null ||
          (params = params(quantizedHebbianNumericLearningClippedWeightsMSNWithConverters, name)) != null ||
          (params = params(quantizedBinaryAndDoublesLearningMsnWithConverter, name)) != null ||
          (params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null) {
        QuantizedValueToSpikeTrainConverter valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
        QuantizedSpikeTrainToValueConverter spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
        if (params.containsKey("iConv")) {
          switch (params.get("iConv")) {
            case "unif":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new QuantizedUniformValueToSpikeTrainConverter();
              }
              break;
            case "unif_mem":
              if (params.containsKey("iFreq")) {
                valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter(Double.parseDouble(params.get("iFreq")));
              } else {
                valueToSpikeTrainConverter = new QuantizedUniformWithMemoryValueToSpikeTrainConverter();
              }
              break;
          }
        }
        if (params.containsKey("oConv")) {
          switch (params.get("oConv")) {
            case "avg":
              if (params.containsKey("oFreq")) {
                spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
              } else {
                spikeTrainToValueConverter = new QuantizedAverageFrequencySpikeTrainToValueConverter();
              }
              break;
            case "avg_mem":
              if (params.containsKey("oFreq")) {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")),
                      Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Double.parseDouble(params.get("oFreq")));
                }
              } else {
                if (params.containsKey("oMem")) {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter(Integer.parseInt(params.get("oMem")));
                } else {
                  spikeTrainToValueConverter = new QuantizedMovingAverageSpikeTrainToValueConverter();
                }
              }
              break;
          }
        }
        if ((params = params(quantizedMsnWithConverter, name)) != null) {
          return new QuantizedMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedNumericLearningMsnWithConverter, name)) != null) {
          return new QuantizedNumericLearningMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedHebbianNumericLearningMsnWithConverter, name)) != null) {
          return new QuantizedHebbianNumericLearningMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedNumericLearningFixedPoolMsnWithConverter, name)) != null) {
          return new QuantizedNumericLearningFixedPoolMSNWithConverters(
              Integer.parseInt(params.get("nRules")),
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedHebbianNumericLearningWeightsMSNWithConverters, name)) != null) {
          return new QuantizedHebbianNumericLearningWeightsMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedBinaryAndDoublesLearningMsnWithConverterAndTuning, name)) != null) {
          return new QuantizedBinaryAndDoublesLearningMSNWithConvertersAndRulesTuning(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedHebbianNumericLearningClippedWeightsMSNWithConverters, name)) != null) {
          return new QuantizedHebbianNumericLearningClippedWeightsMSNWithConverters(
              Double.parseDouble(params.get("ratio")),
              Integer.parseInt(params.get("nLayers")),
              neuronBuilder,
              Double.parseDouble(params.get("maxWeight")),
              valueToSpikeTrainConverter,
              spikeTrainToValueConverter
          );
        }
        if ((params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null ||
            (params = params(quantizedBinaryAndDoublesLearningMsnWithConverter, name)) != null) {
          double[] symmetricParams = {Double.parseDouble(params.get("symmParam1")), Double.parseDouble(params.get("symmParam2")),
              Double.parseDouble(params.get("symmParam3")), Double.parseDouble(params.get("symmParam4"))};
          double[] asymmetricParams = {Double.parseDouble(params.get("asymmParam1")), Double.parseDouble(params.get("asymmParam2")),
              Double.parseDouble(params.get("asymmParam3")), Double.parseDouble(params.get("asymmParam4"))};
          if ((params = params(quantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters, name)) != null) {
            return new QuantizedNumericLearningWithFixedRuleValuesAnd0WeightsMSNWithConverters(
                Double.parseDouble(params.get("ratio")),
                Integer.parseInt(params.get("nLayers")),
                neuronBuilder,
                valueToSpikeTrainConverter,
                spikeTrainToValueConverter,
                symmetricParams,
                asymmetricParams
            );
          }
          if ((params = params(quantizedBinaryAndDoublesLearningMsnWithConverter, name)) != null) {
            return new QuantizedBinaryAndDoublesLearningMSNWithConverters(
                Double.parseDouble(params.get("ratio")),
                Integer.parseInt(params.get("nLayers")),
                neuronBuilder,
                valueToSpikeTrainConverter,
                spikeTrainToValueConverter,
                symmetricParams,
                asymmetricParams
            );
          }
        }
      }
    }
    //misc
    if ((params = params(spikingFunctionGrid, name)) != null) {
      return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    if ((params = params(spikingQuantizedFunctionGrid, name)) != null) {
      return new SpikingQuantizedFunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    return null;
  }

}
