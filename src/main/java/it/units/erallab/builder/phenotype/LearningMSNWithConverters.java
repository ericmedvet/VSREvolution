package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.controllers.snn.*;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.stv.SpikeTrainToValueConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.converters.vts.ValueToSpikeTrainConverter;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.STDPLearningRule;
import it.units.erallab.hmsrobots.core.controllers.snn.learning.SymmetricAntiHebbianLearningRule;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

public class LearningMSNWithConverters implements PrototypedFunctionBuilder<List<STDPLearningRule>, TimedRealFunction> {

  private final double innerLayerRatio;
  private final int nOfInnerLayers;
  private final BiFunction<Integer, Integer, SpikingFunction> neuronBuilder;
  private final ValueToSpikeTrainConverter valueToSpikeTrainConverter;
  private final SpikeTrainToValueConverter spikeTrainToValueConverter;

  public LearningMSNWithConverters(double innerLayerRatio, int nOfInnerLayers, BiFunction<Integer, Integer, SpikingFunction> neuronBuilder, ValueToSpikeTrainConverter valueToSpikeTrainConverter, SpikeTrainToValueConverter spikeTrainToValueConverter) {
    this.innerLayerRatio = innerLayerRatio;
    this.nOfInnerLayers = nOfInnerLayers;
    this.neuronBuilder = neuronBuilder;
    this.valueToSpikeTrainConverter = valueToSpikeTrainConverter;
    this.spikeTrainToValueConverter = spikeTrainToValueConverter;
  }

  private int[] innerNeurons(int nOfInputs, int nOfOutputs) {
    int[] innerNeurons = new int[nOfInnerLayers];
    int centerSize = (int) Math.max(2, Math.round(nOfInputs * innerLayerRatio));
    if (nOfInnerLayers > 1) {
      for (int i = 0; i < nOfInnerLayers / 2; i++) {
        innerNeurons[i] = nOfInputs + (centerSize - nOfInputs) / (nOfInnerLayers / 2 + 1) * (i + 1);
      }
      for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
        innerNeurons[i] = centerSize + (nOfOutputs - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
      }
    } else if (nOfInnerLayers > 0) {
      innerNeurons[0] = centerSize;
    }
    return innerNeurons;
  }

  @Override
  public Function<List<STDPLearningRule>, TimedRealFunction> buildFor(TimedRealFunction function) {
    return values -> {
      int nOfInputs = function.getInputDimension();
      int nOfOutputs = function.getOutputDimension();
      int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs);
      int nOfWeights = MultilayerSpikingNetwork.countWeights(nOfInputs, innerNeurons, nOfOutputs);
      if (nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for learning rules: %d expected, %d found",
            nOfWeights,
            values.size()
        ));
      }
      IndependentFactory<List<Double>> weightsFactory = new FixedLengthListFactory<>(nOfWeights, new UniformDoubleFactory(-1d, 1d));
      return new LearningMultilayerSpikingNetworkWithConverters(
          nOfInputs,
          innerNeurons,
          nOfOutputs,
          weightsFactory.build(new Random()).stream().mapToDouble(Double::doubleValue).toArray(),
          values.toArray(new STDPLearningRule[0]),
          neuronBuilder,
          valueToSpikeTrainConverter,
          spikeTrainToValueConverter
      );
    };
  }

  @Override
  public List<STDPLearningRule> exampleFor(TimedRealFunction function) {
    return Collections.nCopies(
        MultilayerSpikingNetwork.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDimension(),
                innerNeurons(function.getInputDimension(), function.getOutputDimension()),
                function.getOutputDimension())
        ),
        new SymmetricAntiHebbianLearningRule()
    );
  }

}
