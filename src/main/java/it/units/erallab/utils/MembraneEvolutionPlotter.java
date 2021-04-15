package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.snn.IzhikevicNeuron;
import it.units.erallab.hmsrobots.core.controllers.snn.LIFNeuronWithHomeostasis;
import it.units.erallab.hmsrobots.core.controllers.snn.SpikingNeuron;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;

public class MembraneEvolutionPlotter {

  public static void plotMembranePotentialEvolution(SpikingNeuron spikingNeuron, int width, int height) {
    new SwingWrapper<>(getMembranePotentialEvolutionChart(spikingNeuron, width, height)).displayChart();
  }

  public static void plotMembranePotentialEvolutionWithInputSpikes(SpikingNeuron spikingNeuron, int width, int height) {
    new SwingWrapper<>(getMembranePotentialEvolutionWithInputSpikesChart(spikingNeuron, width, height)).displayChart();
  }

  public static void plotMembranePotentialAndRecoveryEvolutionWithInputSpikes(IzhikevicNeuron izhikevicNeuron, int width, int height) {
    new SwingWrapper<>(getMembraneAndRecoveryPotentialEvolutionWithInputSpikesChart(izhikevicNeuron, width, height)).displayChart();
  }

  public static XYChart getMembranePotentialEvolutionChart(SpikingNeuron spikingNeuron, int width, int height) {
    XYChart chart = new XYChart(width, height);
    chart.getStyler().setXAxisMin(0.0).setXAxisMax(spikingNeuron.getLastEvaluatedTime()).setLegendPosition(Styler.LegendPosition.InsideNW);
    if (!spikingNeuron.getMembranePotentialValues().isEmpty()) {
      XYSeries membranePotentialSeries = chart.addSeries("membrane potential", new ArrayList<>(spikingNeuron.getMembranePotentialValues().keySet()), new ArrayList<>(spikingNeuron.getMembranePotentialValues().values()));
      membranePotentialSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line).setMarker(SeriesMarkers.NONE);
    }
    if (spikingNeuron instanceof LIFNeuronWithHomeostasis) {
      SortedMap<Double, Double> thresholdValues = ((LIFNeuronWithHomeostasis) spikingNeuron).getThresholdValues();
      if (!thresholdValues.isEmpty()) {
        XYSeries thresholdSeries = chart.addSeries("threshold", new ArrayList<>(thresholdValues.keySet()), new ArrayList<>(thresholdValues.values()));
        thresholdSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line).setMarker(SeriesMarkers.NONE);
      }
    } else {
      XYSeries thresholdSeries = chart.addSeries("threshold", List.of(0, spikingNeuron.getLastEvaluatedTime()), List.of(spikingNeuron.getThresholdPotential(), spikingNeuron.getThresholdPotential()));
      thresholdSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line).setMarker(SeriesMarkers.NONE);
    }
    return chart;
  }

  public static XYChart getMembranePotentialEvolutionWithInputSpikesChart(SpikingNeuron spikingNeuron, int width, int height) {
    XYChart chart = getMembranePotentialEvolutionChart(spikingNeuron, width, height);
    if (!spikingNeuron.getInputSpikesValues().isEmpty()) {
      XYSeries inputSpikesSeries = chart.addSeries("input", new ArrayList<>(spikingNeuron.getInputSpikesValues().keySet()), new ArrayList<>(spikingNeuron.getInputSpikesValues().values()));
      inputSpikesSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter).setMarker(SeriesMarkers.DIAMOND);
    }
    return chart;
  }

  public static XYChart getMembraneAndRecoveryPotentialEvolutionWithInputSpikesChart(IzhikevicNeuron izhikevicNeuron, int width, int height) {
    XYChart chart = getMembranePotentialEvolutionWithInputSpikesChart(izhikevicNeuron, width, height);
    if (!izhikevicNeuron.getMembraneRecoveryValues().isEmpty()) {
      XYSeries membraneRecoverySeries = chart.addSeries("membrane recovery", new ArrayList<>(izhikevicNeuron.getMembraneRecoveryValues().keySet()), new ArrayList<>(izhikevicNeuron.getMembraneRecoveryValues().values()));
      membraneRecoverySeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line).setMarker(SeriesMarkers.NONE);
    }
    return chart;
  }

}
