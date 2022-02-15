package it.units.erallab.builder.function;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class PruningMLP extends MLP {

  private final PruningMultiLayerPerceptron.Context context;
  private final PruningMultiLayerPerceptron.Criterion criterion;

  public PruningMLP(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      PruningMultiLayerPerceptron.Context context,
      PruningMultiLayerPerceptron.Criterion criterion
  ) {
    super(activationFunction);
    this.context = context;
    this.criterion = criterion;
  }

  @Override
  public PrototypedFunctionBuilder<List<Double>, TimedRealFunction> build(Map<String, String> params) {
    double innerLayerRatio = Double.parseDouble(params.get("r"));
    int nOfInnerLayers = Integer.parseInt(params.get("nIL"));
    double pruningTime = Double.parseDouble(params.get("pT"));
    double rate = Double.parseDouble(params.get("pR"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
        return values -> {
          int nOfInputs = function.getInputDimension();
          int nOfOutputs = function.getOutputDimension();
          int[] innerNeurons = innerNeurons(nOfInputs, nOfOutputs, innerLayerRatio, nOfInnerLayers);
          int nOfWeights = MultiLayerPerceptron.countWeights(nOfInputs, innerNeurons, nOfOutputs);
          if (nOfWeights != values.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values for weights: %d expected, %d found",
                nOfWeights,
                values.size()
            ));
          }
          return new PruningMultiLayerPerceptron(
              activationFunction,
              nOfInputs,
              innerNeurons,
              nOfOutputs,
              values.stream().mapToDouble(d -> d).toArray(),
              pruningTime,
              context,
              criterion,
              rate
          );
        };
      }

      @Override
      public List<Double> exampleFor(TimedRealFunction function) {
        return Collections.nCopies(
            MultiLayerPerceptron.countWeights(
                MultiLayerPerceptron.countNeurons(
                    function.getInputDimension(),
                    innerNeurons(
                        function.getInputDimension(),
                        function.getOutputDimension(),
                        innerLayerRatio,
                        nOfInnerLayers
                    ),
                    function.getOutputDimension()
                )
            ),
            0d
        );
      }
    };
  }


}
